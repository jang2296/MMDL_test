"""CPU workflow/payload counterexamples and own-data checks, never GPU certification."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fine_tuning import audit, bridge_evaluation, circuit_bridge, evaluate, run_bridge_experiment as experiment, run_control
from fine_tuning.modules import ARMS, BASE_ID, BASE_REVISION
from fine_tuning.tests import test_controller as fixtures
from fine_tuning.tests.test_bridge_smoke import historical_processor_fixture

ROOT = Path(experiment.__file__).parent
APPROVAL = ROOT/"runs/recovered_graph_pool32_child_approval_v2/proof"
POOL = ROOT/"generated/graph_cs_bridge_split_pool32_v2"


def evaluation_fixture(root):
    """Authored short response: validates bookkeeping, not measured model answers."""
    cohort, output, protocol = root/"cohort", root/"result", root/"protocol.json"
    record = {"id": "authored-example", "subject": "example", "question_type": "multiple-choice",
              "question": "Authored test", "options": ["first", "second"], "image_paths": ["authored.png"]}
    gold = {"id": record["id"], "answer": "A"}
    audit.write_rows(cohort/"inputs.jsonl", [record])
    audit.write_rows(cohort/"gold.jsonl", [gold])
    audit.dump(cohort/"manifest.json", {"status":"COHORT_FROZEN", "training_allowed":False, "count":1,
        "inputs_sha256":audit.sha(cohort/"inputs.jsonl"), "gold_sha256":audit.sha(cohort/"gold.jsonl")})
    protocol.write_bytes((ROOT/"protocol.json").read_bytes())
    scored = evaluate.score_with_parser("Final answer: A.", "multiple-choice", record["options"], "A",
                                         "team-final-answer-v8", "stop", question=record["question"])
    row = {"id":record["id"], "subject":record["subject"], "status":"COMPLETED", "raw_response":"Final answer: A.",
        "finish_reason":"stop", "seed":evaluate.sample_seed(record["id"]), "seconds":1.,
        "input_sha256":"a"*64, "prompt_token_ids":[1], "output_token_ids":[2], "input_tokens":1,"output_tokens":1,
        "image_grid_thw":[[1,2,2]], "processed_image_sizes":[[32,32]], **scored}
    audit.write_rows(output/"predictions.jsonl", [row])
    summary = {"status":"EVALUATED", "model_id":BASE_ID, "model_revision":BASE_REVISION, "checkpoint":"F0-frozen",
        "cohort_sha256":audit.sha(cohort/"manifest.json"), "protocol_sha256":audit.sha(protocol),
        "predictions_sha256":audit.sha(output/"predictions.jsonl"), "completed":1,"denominator":1,
        "correct":1,"accuracy":1., "seconds":1.,"evaluator_source_sha256":audit.sha(Path(evaluate.__file__)),
        "benchmark_derived":True,"training_allowed":False,"public_upload_allowed":False}
    audit.dump(output/"summary.json", summary)
    return output, cohort, protocol, row, summary


class BridgeExperimentTests(unittest.TestCase):
    def test_circuit_result_gate_replays_typed_score_and_rejects_domain_mislabel(self):
        with tempfile.TemporaryDirectory() as directory:
            output,cohort,protocol,row,summary=evaluation_fixture(Path(directory))
            record=audit.read_rows(cohort/"inputs.jsonl")[0]
            record.update(question_type="open",options=[])
            gold={"id":record["id"],"metric_kind":"circuit_qa","target":"A",
                  "circuit_contract_version":circuit_bridge.VERSION}
            audit.write_rows(cohort/"pairing.jsonl",[{"id":record["id"],"view":"qa"}])
            (cohort/"inputs.jsonl").write_text(json.dumps(record)+"\n")
            (cohort/"gold.jsonl").write_text(json.dumps(gold)+"\n")
            manifest=json.loads((cohort/"manifest.json").read_text())
            manifest.update(inputs_sha256=audit.sha(cohort/"inputs.jsonl"),gold_sha256=audit.sha(cohort/"gold.jsonl"),
                evaluation_domain=bridge_evaluation.CIRCUIT_DOMAIN,metric_policy="authored typed circuit fixture",
                circuit_contract_version=circuit_bridge.VERSION,pairing_sha256=audit.sha(cohort/"pairing.jsonl"))
            (cohort/"manifest.json").write_text(json.dumps(manifest))
            row.update(raw_response="A",**evaluate.score_external("A",gold,"stop"))
            (output/"predictions.jsonl").write_text(json.dumps(row)+"\n")
            summary.update(cohort_sha256=audit.sha(cohort/"manifest.json"),
                predictions_sha256=audit.sha(output/"predictions.jsonl"),benchmark_derived=False,
                evaluation_domain=manifest["evaluation_domain"],metric_policy=manifest["metric_policy"])
            (output/"summary.json").write_text(json.dumps(summary))
            self.assertEqual(1,experiment.result_gate(output,cohort,protocol)[0]["correct"])
            for field,value in (("metric_kind","qa"),("circuit_contract_version","old")):
                changed=dict(gold,**{field:value})
                (cohort/"gold.jsonl").write_text(json.dumps(changed)+"\n")
                manifest["gold_sha256"]=audit.sha(cohort/"gold.jsonl")
                (cohort/"manifest.json").write_text(json.dumps(manifest))
                summary["cohort_sha256"]=audit.sha(cohort/"manifest.json")
                (output/"summary.json").write_text(json.dumps(summary))
                with self.assertRaisesRegex(ValueError,"Circuit gold"):
                    experiment.result_gate(output,cohort,protocol)
            (cohort/"gold.jsonl").write_text(json.dumps(gold)+"\n")
            manifest["gold_sha256"]=audit.sha(cohort/"gold.jsonl")
            (cohort/"manifest.json").write_text(json.dumps(manifest))
            summary["cohort_sha256"]=audit.sha(cohort/"manifest.json")
            row["raw_response"]="B"
            (output/"predictions.jsonl").write_text(json.dumps(row)+"\n")
            summary["predictions_sha256"]=audit.sha(output/"predictions.jsonl")
            (output/"summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,"Saved score"):
                experiment.result_gate(output,cohort,protocol)
            row["raw_response"]="A"
            (output/"predictions.jsonl").write_text(json.dumps(row)+"\n")
            summary.update(predictions_sha256=audit.sha(output/"predictions.jsonl"),
                evaluation_domain=bridge_evaluation.DOMAIN)
            (output/"summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,"metric policy"):
                experiment.result_gate(output,cohort,protocol)

    def test_historical_four_approval_fixtures_are_paired_and_cannot_escape(self):
        reference = None
        for condition in experiment.CONDITIONS:
            with self.assertRaisesRegex(ValueError, "approval/source/processor identity mismatch"):
                experiment.bridge.inputs(APPROVAL, POOL, ROOT/"bridge_training_protocol_v1.json", condition)
            with historical_processor_fixture():
                rows, _manifest, _records, contract, frozen = experiment.bridge.inputs(
                    APPROVAL, POOL, ROOT/"bridge_training_protocol_v1.json", condition)
            actual = [(r["root_id"],r["family_id"],r["view"]) for r in rows]
            if reference is not None:
                self.assertEqual(reference, actual)
            reference = actual
            self.assertEqual((len(rows),len({r["root_id"] for r in rows})), (84,28))
            self.assertEqual(3145728,contract["episode_pixels"])
            self.assertTrue(all(audit.sha(Path(p)) == h for p,h in frozen.items()))
        with self.assertRaisesRegex(ValueError,"Unknown"):
            experiment.bridge.inputs(APPROVAL,POOL,ROOT/"bridge_training_protocol_v1.json","../A3")

    def test_general_arm_verifier_and_wrong_group_payloads(self):
        for arm, groups in ARMS.items():
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                parent, rows, records, manifest, cfg, _protocol, checkpoint = fixtures.ControllerTests().pilot_reuse_fixture(Path(directory))
                metadata = json.loads((checkpoint/"metadata.json").read_text())
                metadata.update(arm=arm)
                metadata["exposure"]["visual_tokens"] = 1600
                metadata["inventory"] = {"arm":arm,"trainable":[{"name":g,"group":g} for g in groups]}
                metadata["state_sha256"] = {g:"b"*64 for g in groups}
                if groups == ("M",):
                    for path in (checkpoint/"adapter").iterdir():
                        path.unlink()
                    (checkpoint/"adapter").rmdir()
                    metadata["files"] = {}
                if "M" in groups:
                    (checkpoint/"mergers.pt").write_bytes(b"authored CPU payload, not tensors")
                    metadata["files"]["mergers.pt"] = audit.sha(checkpoint/"mergers.pt")
                updates = audit.read_rows(checkpoint.parent/"updates.jsonl")
                for row in updates:
                    for key in ("gradient_l2","sampled_update_l1"):
                        row[key] = {g:1. for g in groups}
                (checkpoint.parent/"updates.jsonl").write_text("".join(json.dumps(row)+"\n" for row in updates))
                (checkpoint/"metadata.json").write_text(json.dumps(metadata))
                self.assertEqual(arm,run_control.verify_arm_training(parent/"pilot_FL",rows,records,manifest,cfg,metadata,arm)["arm"])
                forbidden = checkpoint/"adapter" if groups == ("M",) else checkpoint/"unlisted.pt"
                if groups == ("M",):
                    forbidden.mkdir()
                else:
                    forbidden.write_bytes(b"extra unlisted payload")
                with self.assertRaisesRegex(ValueError,"unlisted|wrong-arm"):
                    run_control.verify_arm_training(parent/"pilot_FL",rows,records,manifest,cfg,metadata,arm)

    def test_missing_observed_fields_boolean_scores_and_score_replay_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            output,cohort,protocol,row,summary = evaluation_fixture(Path(directory))
            self.assertEqual(1,experiment.result_gate(output,cohort,protocol)[0]["correct"])
            for bad in (*experiment.runner.PAIRED_INPUT_FIELDS,"int_correct","raw_wrong","wrong_tokens"):
                changed, saved = copy.deepcopy(row), dict(summary)
                if bad in experiment.runner.PAIRED_INPUT_FIELDS:
                    changed.pop(bad)
                elif bad == "int_correct":
                    changed["correct"] = 1
                    saved.update(correct=0,accuracy=0.)
                elif bad == "raw_wrong":
                    changed["raw_response"] = "Final answer: B."
                else:
                    changed["output_tokens"] = 2
                (output/"predictions.jsonl").write_text(json.dumps(changed)+"\n")
                saved["predictions_sha256"] = audit.sha(output/"predictions.jsonl")
                (output/"summary.json").write_text(json.dumps(saved))
                with self.subTest(bad=bad),self.assertRaises(ValueError):
                    experiment.result_gate(output,cohort,protocol)

    def test_exact_checkpoint_identity_and_source_change_are_not_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output,cohort,protocol,_row,summary = evaluation_fixture(root)
            checkpoint = root/"checkpoint"
            audit.dump(checkpoint/"metadata.json",{"authored":"notweights"})
            summary.update(checkpoint=str(checkpoint),checkpoint_metadata_sha256="wrong")
            (output/"summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,"checkpoint"):
                experiment.result_gate(output,cohort,protocol,checkpoint)
            summary.update(checkpoint="F0-frozen",checkpoint_metadata_sha256=None,
                           reused_completed=1,parent_evaluator_source_sha256="missing")
            (output/"summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,"prefix"):
                experiment.result_gate(output,cohort,protocol,evaluator_source=Path(evaluate.__file__))
            summary.pop("reused_completed")
            old = root/"changed_evaluator.py"
            old.write_text(Path(evaluate.__file__).read_text().replace('master=3407','master=3408'))
            summary["evaluator_source_sha256"] = audit.sha(old)
            (output/"summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,"generation changed"):
                experiment.result_gate(output,cohort,protocol,evaluator_source=old)

    def test_grouping_uses_shared_root_condition_and_family_not_independent_rows(self):
        cohort = ROOT/"generated/graph_cs_bridge_development_cohort_v1"
        ledger = audit.read_rows(cohort/"pairing.jsonl")
        rows = [{"id":row["id"],"correct":False,"metric_kind":row["view"]} for row in ledger]
        after = [dict(row,correct=i==0) for i,row in enumerate(rows)]
        result = experiment.graph_groups(cohort,rows,after)
        self.assertEqual((4,2,2),tuple(len(result[k]) for k in ("condition","root_id","family_id")))
        self.assertEqual(1,sum(group["qa"]["gain"] for group in result["root_id"].values()))
        with self.assertRaises(ValueError):
            experiment.graph_groups(cohort,rows,list(reversed(after)))

    def test_live_previous_process_is_rejected_before_inputs_or_gpu(self):
        controller = Mock()
        with patch.object(experiment,"process_gone",return_value=False):
            with self.assertRaisesRegex(ValueError,"still live"):
                experiment.run(SimpleNamespace(previous_pid=1,previous_birth="1"),controller)
        controller.stage.assert_not_called()

    def test_budget_stop_precedes_gpu_with_fresh_baseline_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit.dump(root/"model/config.json",{"vision_config":{"spatial_merge_size":2}})
            audit.dump(root/"reference/metadata.json",{"arm":"FL","base_id":BASE_ID,"base_revision":BASE_REVISION,
                "epoch_exposure_complete":True,"planned_records":84,"exposure":{"records":84,"updates":6},
                "consumed_order_sha256":"same","record_order_sha256":"same"})
            args = SimpleNamespace(previous_pid=1,previous_birth="1",previous_controller=root,arm="FL",
                protocol=ROOT/"protocol.json",reference_checkpoint=root/"reference",model_path=root/"model",
                graph_cohort=root/"graph",mmmu_cohort=root/"mmmu",baseline_mmmu=None,baseline_evaluator_source=None,
                baseline_parent_evaluator_source=None,reuse_a3_controller=None,training_cap_seconds=900,evaluation_cap_seconds=3600)
            controller = Mock(output=root/"output")
            controller.remaining.return_value=120
            contract={"sequence_budget":4096,"min_pixels":262144,"max_pixels":1048576,"episode_pixels":3145728}
            audit.dump(root/"graph/manifest.json", {"count":24,"root_count":2,
                "evaluation_domain":bridge_evaluation.DOMAIN})
            datasets={c:([{"root_id":f"root{i//3}"} for i in range(84)],root/"manifest",root/"records",contract)
                      for c in experiment.CONDITIONS}
            with patch.object(experiment,"process_gone",return_value=True), \
                    patch.object(experiment.runner,"closed_controller",return_value=({},[],{})), \
                    patch.object(experiment,"training_bundle",return_value=(datasets,{})), \
                    patch.object(experiment,"require_idle_gpu") as idle:
                result=experiment.run(args,controller)
            self.assertEqual("BUDGET_STOP",result["status"])
            self.assertFalse(result["GPU_started"])
            controller.stage.assert_not_called()
            idle.assert_not_called()

    def test_authored_full_workflow_runs_every_condition_from_f0_on_common_inputs(self, circuit=False):
        """Mocked subprocesses exercise routing only, not training or GPU scores."""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            contract={"sequence_budget":4096,"min_pixels":262144,"max_pixels":1048576,"episode_pixels":3145728}
            audit.dump(root/"model/config.json",{"vision_config":{"spatial_merge_size":2}})
            reference={"arm":"FL","base_id":BASE_ID,"base_revision":BASE_REVISION,
                "epoch_exposure_complete":True,"planned_records":84,"exposure":{"records":84,"updates":6},
                "consumed_order_sha256":"same","record_order_sha256":"same"}
            audit.dump(root/"reference/metadata.json",reference)
            datasets={}
            for condition in experiment.CONDITIONS:
                rows=[{"id":f"{condition}-{i}","root_id":f"root{i//3}","view":("qa","structure","grounded")[i%3]}
                      for i in range(84)]
                datasets[condition]=(rows,root/f"{condition}_manifest",root/f"{condition}_records",contract)
            args=SimpleNamespace(previous_pid=1,previous_birth="1",previous_controller=root,arm="FL",
                protocol=ROOT/"protocol.json",reference_checkpoint=root/"reference",model_path=root/"model",
                image_root=root/"images",approval=root/"approval",input_protocol=ROOT/"bridge_training_protocol_v1.json",
                graph_cohort=root/"graph",mmmu_cohort=root/"mmmu",baseline_mmmu=None,baseline_evaluator_source=None,
                baseline_parent_evaluator_source=None,reuse_a3_controller=None,training_cap_seconds=900,evaluation_cap_seconds=3600)
            for name,count in (("graph",24),("mmmu",60)):
                rows=[{"id":f"{name}-{i}","subject":"authored"} for i in range(count)]
                audit.write_rows(root/name/"inputs.jsonl",rows)
                audit.dump(root/name/"manifest.json",{"count":count,"root_count":2,
                    "evaluation_domain":bridge_evaluation.CIRCUIT_DOMAIN if circuit else bridge_evaluation.DOMAIN})
            commands=[]
            def stage(name,command,maximum_seconds):
                commands.append(command)
                output=Path(command[command.index("--output")+1]) if "--output" in command else None
                if "--verify-checkpoint" in command:
                    checkpoint=command[command.index("--verify-checkpoint")+1]
                    (root/"output"/(name+".log")).write_text(json.dumps({"status":"RELOAD_VERIFIED","checkpoint":checkpoint})+"\n")
                elif "fine_tuning.train" in command:
                    condition=name[:2]
                    audit.dump(output/"checkpoint/metadata.json",dict(reference,termination="EPOCH_COMPLETE",files={}))
                    audit.write_rows(output/"consumed_ids.jsonl",datasets[condition][0])
                    audit.write_rows(output/"updates.jsonl",[{"authored":"notGPU"}])
                else:
                    cohort=Path(command[command.index("--cohort")+1])
                    records=audit.read_rows(cohort/"inputs.jsonl")
                    rows=[dict(record,status="COMPLETED",correct=True,metric_kind="qa",input_sha256="same",
                        prompt_token_ids=[1],image_grid_thw=[[1,2,2]],processed_image_sizes=[[32,32]],seed=1) for record in records]
                    audit.write_rows(output/"predictions.jsonl",rows)
                    checkpoint=command[command.index("--checkpoint")+1] if "--checkpoint" in command else None
                    audit.dump(output/"summary.json",{"status":"EVALUATED","model_id":BASE_ID,"model_revision":BASE_REVISION,
                        "checkpoint":checkpoint or "F0-frozen","cohort_sha256":audit.sha(cohort/"manifest.json"),
                        "protocol_sha256":audit.sha(args.protocol),"denominator":len(rows),"completed":len(rows),
                        "predictions_sha256":audit.sha(output/"predictions.jsonl")})
            controller=Mock(output=root/"output")
            controller.remaining.return_value=999999
            controller.stage.side_effect=stage
            with patch.object(experiment,"process_gone",return_value=True), \
                    patch.object(experiment.runner,"closed_controller",return_value=({},[],{})), \
                    patch.object(experiment,"training_bundle",return_value=(datasets,{})), \
                    patch.object(experiment,"require_idle_gpu"), patch.object(experiment,"checkpoint_layout"), \
                    patch.object(experiment,"verify_arm_training",return_value=dict(reference,files={})), \
                    patch.object(experiment.bridge,"verify_processor_exposure"), patch.object(experiment,"result_gate"), \
                    patch.object(experiment,"graph_groups",return_value={"authored":"routing only"}):
                result=experiment.run(args,controller)
            self.assertEqual(f"BOUNDED_{'CIRCUIT' if circuit else 'GRAPH'}_FIXED_MODULE_A0_A3_EVALUATED",result["status"])
            self.assertEqual(18,len(commands))
            self.assertEqual(set(experiment.CONDITIONS),set(result["results"]))
            self.assertEqual({"A1-A0","A2-A0","A3-A2","A3-A1"},set(result["contrasts"]))
            for command in commands:
                if "fine_tuning.train" in command and "--verify-checkpoint" not in command:
                    self.assertNotIn("--checkpoint",command)
                    self.assertEqual("FL",command[command.index("--arm")+1])
                if "fine_tuning.evaluate" in command:
                    self.assertIn(command[command.index("--cohort")+1],{str(args.graph_cohort),str(args.mmmu_cohort)})
            self.assertFalse(result["full_professional_G7_completed"])
            external = "circuit" if circuit else "graph"
            self.assertTrue(all(set(r)=={external,"mmmu"} for r in result["results"].values()))
            declaration=json.loads((controller.output/"input_freeze.json").read_text())
            self.assertEqual((84,28,6,24,2),tuple(declaration[k] for k in (
                "records_per_condition","roots_per_condition","updates_per_condition",
                "common_bridge_evaluation_records","independent_bridge_roots")))
            audit.dump(controller.output/"status.json",{"status":result["status"]})
            audit.dump(controller.output/"source_freeze.json",{"fixture":"not GPU"})
            audit.write_rows(controller.output/"journal.jsonl",[{"stage":"controller","status":result["status"]}])
            self.assertEqual(result["status"],experiment.runner.closed_controller(controller.output)[0]["status"])

    def test_authored_circuit_workflow_uses_circuit_names_and_closed_status(self):
        self.test_authored_full_workflow_runs_every_condition_from_f0_on_common_inputs(circuit=True)


if __name__ == "__main__":
    unittest.main()
