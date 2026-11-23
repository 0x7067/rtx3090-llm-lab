"""Exercise restoration failures and baseline reuse without a live cluster."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import baseline_trial

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
OWNER = "bench.denguinho.org/sparsekv-owner"
RECONCILE = "kustomize.toolkit.fluxcd.io/reconcile"

MOCK = '''import json, os, sys
from pathlib import Path
p=Path(os.environ['MOCK_STATE']); s=json.loads(p.read_text())
tool=Path(sys.argv[0]).name
args=sys.argv[1:]
s['calls'].append([tool,*args])
rc=0
if tool=='kubectl':
    if 'get' in args:
        s['reads']+=1
        if s['scenario']=='read-timeout' and s['reads']==1: rc=1
        else: print(json.dumps(s['deployment']))
    elif 'patch' in args:
        s['patches']+=1
        if s['scenario']=='patch-before-timeout' and s['patches']==1: rc=1
        else:
            d=s['deployment']
            for op in json.loads(args[args.index('-p')+1]):
                node=d; keys=[k.replace('~1','/').replace('~0','~') for k in op['path'].split('/')[1:]]
                for key in keys[:-1]: node=node[key]
                key=keys[-1]
                if op['op']=='test': assert node[key]==op['value']
                elif op['op']=='remove': del node[key]
                else: node[key]=op['value']
            if s['scenario']=='patch-after-timeout' and s['patches']==1: rc=1
    elif 'exec' in args: print('42')
p.write_text(json.dumps(s))
sys.exit(rc)
'''


class RestorationTests(unittest.TestCase):
    def restore(self, scenario):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Work on an independent copy of the recorded production config.
            shutil.copy2(BENCH / "production/deployment.json", root / "deployment.json")
            deployment = json.loads((root / "deployment.json").read_text())
            deployment["metadata"].update(uid="original", generation=1, annotations={OWNER: "ours", RECONCILE: "disabled"})
            deployment["spec"]["replicas"] = 0
            deployment["status"] = {"availableReplicas": 1, "observedGeneration": 1}
            if scenario == "another-owner":
                deployment["metadata"]["annotations"][OWNER] = "another"
            if scenario == "another-uid":
                deployment["metadata"]["uid"] = "replacement"
            state = root / "state.json"
            state.write_text(json.dumps(dict(scenario=scenario, deployment=deployment, reads=0, patches=0, calls=[])))
            (root / "claim.json").write_text(json.dumps(dict(owner="ours", uid="original", replicas=1, reconcile=None)))
            (root / ".claimed").touch()
            mockbin = root / "bin"
            mockbin.mkdir()
            for name in ["kubectl", "sudo", "sleep", "uv"]:
                program = mockbin / name
                program.write_text(f"#!{sys.executable}\n" + MOCK)
                program.chmod(0o755)
            result = subprocess.run(
                ["bash", str(HERE / "orchestrate.sh"), "--restore", str(root)],
                env={**os.environ, "PATH": str(mockbin) + os.pathsep + os.environ["PATH"], "MOCK_STATE": str(state)},
                capture_output=True, text=True, timeout=15,
            )
            return result, json.loads(state.read_text()), (root / "restored.json").exists()

    def test_initial_api_read_timeout_recovers(self):
        result, state, restored = self.restore("read-timeout")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(restored)
        self.assertGreaterEqual(state["reads"], 3)
        self.assertEqual(state["deployment"]["spec"]["replicas"], 1)

    def test_patch_timeouts_recheck_state_before_retry(self):
        for scenario, patches in [("patch-before-timeout", 2), ("patch-after-timeout", 1)]:
            with self.subTest(scenario=scenario):
                result, state, restored = self.restore(scenario)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(restored)
                self.assertEqual(state["patches"], patches)
                self.assertNotIn(OWNER, state["deployment"]["metadata"]["annotations"])
                self.assertNotIn(RECONCILE, state["deployment"]["metadata"]["annotations"])

    def test_uid_and_owner_conflicts_stop_before_mutations(self):
        for scenario in ["another-owner", "another-uid"]:
            with self.subTest(scenario=scenario):
                result, state, restored = self.restore(scenario)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(restored)
                self.assertEqual(state["patches"], 0)
                self.assertTrue(all(call[0] == "kubectl" for call in state["calls"]))


class BaselineTests(unittest.TestCase):
    def test_legacy_chain_and_explicit_metadata_match_saved_hashes(self):
        expected = json.loads((BENCH / "run-2/claim.json").read_text())["owner"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["run-2", "run-3"]:
                shutil.copytree(BENCH / name / "baseline", root / name / "baseline")
                for filename in ["claim.json", "baseline-exit"]:
                    shutil.copy2(BENCH / name / filename, root / name / filename)
            # Resolve the relocated legacy run-3 -> run-2 provenance.
            shutil.copy2(BENCH / "run-3/baseline-source.txt", root / "run-3/baseline-source.txt")
            (root / "run-3/baseline-source.txt").write_text(str(root / "old-worktree/run-2"))
            self.assertEqual(baseline_trial.resolve(root / "run-3"), expected)
            shutil.copytree(root / "run-3", root / "run-4")
            (root / "run-4/baseline-source.txt").write_text(str(root / "run-3"))
            self.assertEqual(baseline_trial.resolve(root / "run-4"), expected)
            (root / "run-4/baseline/trial.json").write_text(json.dumps({"trial_id": expected}))
            (root / "run-4/baseline-source.txt").unlink()
            self.assertEqual(baseline_trial.resolve(root / "run-4"), expected)
            # A run owner's identity must fail validation against copied requests.
            wrong = json.loads((BENCH / "run-3/claim.json").read_text())["owner"]
            (root / "run-4/baseline/trial.json").write_text(json.dumps({"trial_id": wrong}))
            with self.assertRaisesRegex(ValueError, "saved request hash"):
                baseline_trial.resolve(root / "run-4")


if __name__ == "__main__":
    unittest.main()
