import sys,unittest,tempfile,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_partitions import check_integrity
from cohort_loader import require_protocol_lock


class PartitionChecks(unittest.TestCase):
    def row(self,id,patient,fold):return {'ecg_id':id,'patient_id':patient,'strat_fold':str(fold),'filename_lr':'records100/00000/'+id+'_lr'}

    def test_repeated_patient_in_different_folds_rejected(self):
        with self.assertRaises(ValueError):check_integrity([self.row('1','p',1),self.row('2','p',10)])

    def test_repeated_patient_within_fold_allowed(self):
        self.assertEqual(check_integrity([self.row('1','p',1),self.row('2','p',1)]),{'p':1})

    def test_waveform_alias_rejected(self):
        rows=[self.row('1','a',1),self.row('2','b',2)];rows[1]['filename_lr']=rows[0]['filename_lr']
        with self.assertRaises(ValueError):check_integrity(rows)

    def test_final_test_requires_matching_frozen_protocol(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);(root/'config').mkdir()
            with self.assertRaises(PermissionError):require_protocol_lock(root)
            p=root/'config/analysis_protocol.json';p.write_text(json.dumps({'status':'draft_not_frozen'}))
            lock=root/'config/protocol_lock.json';lock.write_text(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
            with self.assertRaises(PermissionError):require_protocol_lock(root)
            p.write_text(json.dumps({'status':'frozen'}))
            with self.assertRaises(PermissionError):require_protocol_lock(root)
            lock.write_text(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}));require_protocol_lock(root)


if __name__=='__main__':unittest.main()
