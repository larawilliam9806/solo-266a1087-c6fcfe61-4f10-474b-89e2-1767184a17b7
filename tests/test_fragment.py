"""片段坐标模式测试：定位、反向换算、端到端上传与换参考重验。

全长参考 REF_V1 = GCTTTGCACGT（11 bp）：
  位置  1 2 3 4 5 6 7 8 9 10 11
  碱基  G C T T T G C A C G  T

片段 F1 = TTTG   -> 正向唯一匹配 3-6
片段 F2 = ACGTGC -> 反向唯一匹配 6-11（revcomp = GCACGT）
片段 F3 = AAGC   -> 反向唯一匹配 1-4（revcomp = GCTT），用于左锚边界测试
"""

import io
import os
import random
import sqlite3
import tempfile
import unittest

from app import create_app, services
from app.normalization import (
    ValidationError,
    locate_fragment,
    revcomp,
    validate_and_normalize_fragment,
)

REF_V1 = "GCTTTGCACGT"
REF_V2 = "GCTATGCACGT"          # 第 4 位 T->A
REF_V3 = "ACGTACGTACGT"         # 与 F2 无匹配
REF_V4 = "TTTGCACGTTTT"         # F2 反向唯一匹配 4-9

F1 = "TTTG"                     # 正向 3-6
F2 = "ACGTGC"                   # 反向 6-11
F3 = "AAGC"                     # 反向 1-4


def tsv(text: str) -> bytes:
    return text.encode("utf-8")


def row(pos, ref, alt, gt="0/1", lineno=1):
    return {"lineno": lineno, "pos": pos, "ref": ref, "alt": alt, "gt": gt}


def full_haplotype(genome, pos, ref, alt):
    return genome[: pos - 1] + alt + genome[pos - 1 + len(ref):]


def fragment_haplotype(genome, fstart, fend, strand, pos, ref, alt):
    """把片段坐标变异应用到全长参考上得到的单倍型（黄金对照）。"""
    frag = genome[fstart - 1 : fend]
    if strand == "-":
        frag = revcomp(frag)
    new_frag = frag[: pos - 1] + alt + frag[pos - 1 + len(ref):]
    if strand == "-":
        new_frag = revcomp(new_frag)
    return genome[: fstart - 1] + new_frag + genome[fend:]


# --------------------------------------------------------------------------- #
# 纯函数：片段定位
# --------------------------------------------------------------------------- #
class LocateFragmentTest(unittest.TestCase):
    def test_unique_forward(self):
        self.assertEqual(locate_fragment(F1, REF_V1), (3, 6, "+"))

    def test_unique_reverse(self):
        self.assertEqual(locate_fragment(F2, REF_V1), (6, 11, "-"))
        self.assertEqual(locate_fragment(F3, REF_V1), (1, 4, "-"))

    def test_no_match_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            locate_fragment("AAAA", REF_V1)
        self.assertIn("无匹配", str(ctx.exception))

    def test_longer_than_genome_no_match(self):
        with self.assertRaises(ValidationError) as ctx:
            locate_fragment(REF_V1 + "A", REF_V1)
        self.assertIn("无匹配", str(ctx.exception))

    def test_multiple_forward_matches_rejected(self):
        # TT 在 TTT 重复区匹配第 3、4 位两处
        with self.assertRaises(ValidationError) as ctx:
            locate_fragment("TT", REF_V1)
        msg = str(ctx.exception)
        self.assertIn("不唯一", msg)
        self.assertIn("2 个位置/方向组合", msg)

    def test_palindrome_same_position_both_strands_rejected(self):
        # AT 反向互补仍是 AT：同一位置正/反两个方向都算匹配，组合不唯一
        with self.assertRaises(ValidationError) as ctx:
            locate_fragment("AT", "AATT")
        self.assertIn("不唯一", str(ctx.exception))

    def test_overlapping_matches_counted(self):
        # AAA 在 AAAAA 中正向匹配 1、2、3 三处
        with self.assertRaises(ValidationError) as ctx:
            locate_fragment("AAA", "AAAAA")
        self.assertIn("3 个位置/方向组合", str(ctx.exception))


# --------------------------------------------------------------------------- #
# 纯函数：片段校验 + 换算 + 归一化
# --------------------------------------------------------------------------- #
class FragmentValidateTest(unittest.TestCase):
    def test_forward_snp_and_ins(self):
        rows = [row(1, "T", "TT", lineno=2), row(4, "G", "A", lineno=3)]
        variants, loc = validate_and_normalize_fragment(rows, F1, REF_V1)
        self.assertEqual(loc, (3, 6, "+"))
        # 插入归一化为经典 (2,C,CT)，SNP 平移到全长第 6 位
        keys = {(v["pos"], v["ref"], v["alt"]) for v in variants}
        self.assertEqual(keys, {(2, "C", "CT"), (6, "G", "A")})
        # orig 保留片段坐标原记录
        self.assertEqual(
            (variants[0]["orig_pos"], variants[0]["orig_ref"], variants[0]["orig_alt"]),
            (1, "T", "TT"),
        )

    def test_reverse_snp(self):
        variants, loc = validate_and_normalize_fragment(
            [row(4, "T", "A")], F2, REF_V1
        )
        self.assertEqual(loc, (6, 11, "-"))
        v = variants[0]
        self.assertEqual((v["pos"], v["ref"], v["alt"]), (8, "A", "T"))
        self.assertEqual(v["kind"], "SNP")
        self.assertEqual((v["orig_pos"], v["orig_ref"], v["orig_alt"]), (4, "T", "A"))

    def test_reverse_insertion_anchor_inside_fragment(self):
        variants, _ = validate_and_normalize_fragment(
            [row(2, "C", "CAA")], F2, REF_V1
        )
        v = variants[0]
        self.assertEqual((v["pos"], v["ref"], v["alt"]), (9, "C", "CTT"))
        self.assertEqual(v["kind"], "INS")
        # 单倍型保持
        self.assertEqual(
            full_haplotype(REF_V1, v["pos"], v["ref"], v["alt"]),
            fragment_haplotype(REF_V1, 6, 11, "-", 2, "C", "CAA"),
        )

    def test_reverse_insertion_anchor_borrowed_outside_fragment(self):
        # 插入锚定在片段最后一位（片段坐标 6），正向左锚取自片段外第 5 位 T
        variants, _ = validate_and_normalize_fragment(
            [row(6, "C", "CA")], F2, REF_V1
        )
        v = variants[0]
        self.assertEqual((v["pos"], v["ref"], v["alt"]), (2, "C", "CT"))  # 归一化后
        self.assertEqual(
            full_haplotype(REF_V1, v["pos"], v["ref"], v["alt"]),
            fragment_haplotype(REF_V1, 6, 11, "-", 6, "C", "CA"),
        )

    def test_reverse_deletion_anchor_borrowed_outside_fragment(self):
        # 缺失片段坐标 6 的 C（REF=GC 锚在 5），正向左锚取自片段外第 5 位 T
        variants, _ = validate_and_normalize_fragment(
            [row(5, "GC", "G")], F2, REF_V1
        )
        v = variants[0]
        self.assertEqual((v["pos"], v["ref"], v["alt"]), (5, "TG", "T"))
        self.assertEqual(v["kind"], "DEL")
        self.assertEqual(
            full_haplotype(REF_V1, v["pos"], v["ref"], v["alt"]),
            fragment_haplotype(REF_V1, 6, 11, "-", 5, "GC", "G"),
        )

    def test_reverse_deletion_anchor_inside_fragment(self):
        # F3 反向匹配 1-4；缺失片段坐标 3 的 G（REF=AG 锚在 2），左锚在片段内
        variants, loc = validate_and_normalize_fragment(
            [row(2, "AG", "A")], F3, REF_V1
        )
        self.assertEqual(loc, (1, 4, "-"))
        v = variants[0]
        self.assertEqual((v["pos"], v["ref"], v["alt"]), (1, "GC", "G"))
        self.assertEqual(
            full_haplotype(REF_V1, v["pos"], v["ref"], v["alt"]),
            fragment_haplotype(REF_V1, 1, 4, "-", 2, "AG", "A"),
        )

    def test_no_legal_left_anchor_insertion_rejected(self):
        # F3 反向匹配在参考起点 1-4；片段末端插入换算后无左锚可取
        with self.assertRaises(ValidationError) as ctx:
            validate_and_normalize_fragment([row(4, "C", "CA", lineno=7)], F3, REF_V1)
        msg = str(ctx.exception)
        self.assertIn("第 7 行", msg)
        self.assertIn("左锚", msg)

    def test_no_legal_left_anchor_deletion_rejected(self):
        # 缺失延伸到片段末端（REF=GC 覆盖 3-4），换算后左锚在参考起点之外
        with self.assertRaises(ValidationError) as ctx:
            validate_and_normalize_fragment([row(3, "GC", "G", lineno=3)], F3, REF_V1)
        self.assertIn("第 3 行", str(ctx.exception))
        self.assertIn("左锚", str(ctx.exception))

    def test_reverse_snp_at_fragment_edge_needs_no_anchor(self):
        # SNP 不需要锚，即使片段在参考起点也能换算
        variants, _ = validate_and_normalize_fragment(
            [row(4, "C", "T")], F3, REF_V1
        )
        self.assertEqual(
            (variants[0]["pos"], variants[0]["ref"], variants[0]["alt"]),
            (1, "G", "A"),
        )

    def test_ref_mismatch_against_fragment(self):
        with self.assertRaises(ValidationError) as ctx:
            validate_and_normalize_fragment([row(1, "A", "C", lineno=5)], F1, REF_V1)
        msg = str(ctx.exception)
        self.assertIn("第 5 行", msg)
        self.assertIn("REF 核对失败", msg)
        self.assertIn("片段 1-1 位为 'T'", msg)

    def test_range_beyond_fragment_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            validate_and_normalize_fragment([row(4, "GC", "G", lineno=2)], F1, REF_V1)
        self.assertIn("超出片段长度 4", str(ctx.exception))

    def test_normalized_duplicate_rejected_with_lineno(self):
        rows = [row(1, "T", "TT", lineno=2), row(2, "T", "TT", lineno=3)]
        with self.assertRaises(ValidationError) as ctx:
            validate_and_normalize_fragment(rows, F1, REF_V1)
        msg = str(ctx.exception)
        self.assertIn("第 3 行", msg)
        self.assertIn("第 2 行", msg)
        self.assertIn("同一变异", msg)

    def test_fragment_row_field_validation(self):
        with self.assertRaises(ValidationError):
            validate_and_normalize_fragment([row(1, "T", "TT", gt="0/0")], F1, REF_V1)
        with self.assertRaises(ValidationError):
            validate_and_normalize_fragment([row(1, "t", "TT")], F1, REF_V1)


class FragmentConversionPropertyTest(unittest.TestCase):
    """随机性质测试：片段坐标变异换算+归一化后，全长单倍型必须保持不变。"""

    BASES = "ACGT"

    def random_genome(self, rng, length):
        if rng.random() < 0.5:
            unit = rng.choice(["A", "T", "AT", "TG", "GC"])
            return (unit * (length // len(unit) + 1))[:length]
        return "".join(rng.choice(self.BASES) for _ in range(length))

    def test_haplotype_preserved(self):
        rng = random.Random(20261120)
        checked = 0
        rejected_at_edge = 0
        for _ in range(4000):
            genome = self.random_genome(rng, rng.randint(8, 40))
            i = rng.randrange(len(genome))
            j = min(len(genome), i + rng.randint(3, 12))
            if j <= i:
                continue
            strand_pick = rng.choice(["+", "-"])
            sub = genome[i:j]
            fragment = sub if strand_pick == "+" else revcomp(sub)
            try:
                fstart, fend, strand = locate_fragment(fragment, genome)
            except ValidationError:
                continue  # 非唯一匹配：跳过（唯一定位是换算的前提）
            # 唯一定位必须就是我们取样的位置与方向
            self.assertEqual((fstart, fend, strand), (i + 1, j, strand_pick))

            # 在片段坐标内生成合法变异
            pos = rng.randint(1, len(fragment))
            kind = rng.choice(["SNP", "INS", "DEL"])
            if kind == "DEL" and pos == len(fragment):
                kind = "INS"
            if kind == "SNP":
                ref = fragment[pos - 1]
                alt = rng.choice([b for b in self.BASES if b != ref])
            elif kind == "INS":
                ref = fragment[pos - 1]
                alt = ref + "".join(rng.choice(self.BASES) for _ in range(rng.randint(1, 3)))
            else:
                n = rng.randint(1, len(fragment) - pos)
                ref = fragment[pos - 1 : pos + n]
                alt = fragment[pos - 1]

            try:
                variants, _ = validate_and_normalize_fragment(
                    [row(pos, ref, alt)], fragment, genome
                )
            except ValidationError:
                # 只应发生在「反向 + 锚定延伸到片段末端 + 片段对齐到参考起点」
                self.assertEqual(strand, "-")
                self.assertEqual(fstart, 1)
                rejected_at_edge += 1
                continue

            v = variants[0]
            expected = fragment_haplotype(genome, fstart, fend, strand, pos, ref, alt)
            self.assertEqual(
                full_haplotype(genome, v["pos"], v["ref"], v["alt"]),
                expected,
                f"{(pos, ref, alt)} on fragment({fstart}-{fend},{strand}) of {genome}",
            )
            self.assertEqual(
                (v["orig_pos"], v["orig_ref"], v["orig_alt"]), (pos, ref, alt)
            )
            checked += 1
        self.assertGreater(checked, 1500, "有效随机样本不足，测试可信度不够")
        self.assertGreater(rejected_at_edge, 0, "未覆盖到无左锚拒绝的边界情形")


# --------------------------------------------------------------------------- #
# 端到端：网页上传 / 重传 / 换参考重验
# --------------------------------------------------------------------------- #
class FragmentAppTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        os.environ["DATABASE_PATH"] = self.db_path
        self.app = create_app()
        self.app.config.update(TESTING=True, SECRET_KEY="test")
        self.client = self.app.test_client()

    def tearDown(self):
        os.environ.pop("DATABASE_PATH", None)
        self.tmp.cleanup()

    def upload_ref(self, seq: str):
        return self.client.post(
            "/reference",
            data={"reference": (io.BytesIO(seq.encode()), "ref.txt")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )

    def upload_sample(self, name: str, content: bytes, fragment: bytes | None = None):
        data = {"name": name, "sample": (io.BytesIO(content), f"{name}.tsv")}
        if fragment is not None:
            data["fragment"] = (io.BytesIO(fragment), f"{name}.frag.txt")
        return self.client.post(
            "/samples", data=data, content_type="multipart/form-data",
            follow_redirects=True,
        )

    def db_query(self, sql, args=()):
        with self.app.app_context():
            from app.db import get_db
            return get_db().execute(sql, args).fetchall()

    def dashboard_map(self):
        with self.app.app_context():
            dash = services.get_dashboard()
        return {
            (v["pos"], v["ref"], v["alt"]): v["n_samples"]
            for v in dash["variants"]
        }


class FragmentUploadTest(FragmentAppTestBase):
    def setUp(self):
        super().setUp()
        self.upload_ref(REF_V1)

    def test_reverse_fragment_upload_and_detail(self):
        resp = self.upload_sample(
            "SF", tsv("POS\tREF\tALT\tGT\n4\tT\tA\t0/1\n2\tC\tCAA\t1/1\n"),
            fragment=F2.encode(),
        )
        page = resp.get_data(as_text=True)
        self.assertIn("片段定位：全长 6-11（反向）", page)
        summary = self.dashboard_map()
        self.assertEqual(summary[(8, "A", "T")], 1)
        self.assertEqual(summary[(9, "C", "CTT")], 1)
        self.assertEqual(len(summary), 2)

        # 样本列表显示片段位置与方向
        self.assertIn("片段 6-11", page)

        # 详情页：片段位置、方向与 TSV 原记录
        detail = self.client.get("/variants/8/A>T").get_data(as_text=True)
        self.assertIn("片段 6-11", detail)
        self.assertIn(">4<", detail)          # 原始 POS（片段坐标）
        self.assertIn(">T<", detail)          # 原始 REF
        self.assertIn(">A<", detail)          # 原始 ALT

    def test_forward_fragment_upload(self):
        resp = self.upload_sample("SF", tsv("1\tT\tTT\t0/1\n"), fragment=F1.encode())
        self.assertIn("片段定位：全长 3-6（正向）", resp.get_data(as_text=True))
        self.assertEqual(self.dashboard_map()[(2, "C", "CT")], 1)

    def test_fragment_and_full_length_variants_merge(self):
        # 全长样本与反向片段样本的等价插入归一化为同一规范变异
        self.upload_sample("S1", tsv("3\tT\tTT\t0/1\n"))
        self.upload_sample("SF", tsv("6\tC\tCA\t0/1\n"), fragment=F2.encode())
        summary = self.dashboard_map()
        self.assertEqual(summary[(2, "C", "CT")], 2)
        detail = self.client.get("/variants/2/C>CT").get_data(as_text=True)
        self.assertIn("全长", detail)          # S1 行
        self.assertIn("片段 6-11", detail)     # SF 行
        self.assertIn(">6<", detail)           # SF 原始 POS（片段坐标）
        self.assertIn(">3<", detail)           # S1 原始 POS（全长坐标）

    def test_no_match_fragment_rejected(self):
        resp = self.upload_sample("SF", tsv("1\tA\tC\t0/1\n"), fragment=b"AAAAAA")
        page = resp.get_data(as_text=True)
        self.assertIn("上传失败", page)
        self.assertIn("无匹配", page)
        self.assertEqual(self.db_query("SELECT COUNT(*) c FROM samples")[0]["c"], 0)

    def test_non_unique_fragment_rejected(self):
        resp = self.upload_sample("SF", tsv("1\tT\tC\t0/1\n"), fragment=b"TT")
        page = resp.get_data(as_text=True)
        self.assertIn("上传失败", page)
        self.assertIn("不唯一", page)
        self.assertEqual(self.db_query("SELECT COUNT(*) c FROM samples")[0]["c"], 0)

    def test_row_error_reports_original_lineno(self):
        # 带表头：第 4 行（第 3 条数据）REF 与片段不符
        content = tsv("POS\tREF\tALT\tGT\n1\tT\tTT\t0/1\n4\tG\tA\t0/1\n1\tA\tC\t1/1\n")
        resp = self.upload_sample("SF", content, fragment=F1.encode())
        page = resp.get_data(as_text=True)
        self.assertIn("第 4 行", page)
        self.assertIn("片段 1-1 位为", page)  # 单引号在页面中被 HTML 转义
        self.assertEqual(self.db_query("SELECT COUNT(*) c FROM samples")[0]["c"], 0)

    def test_failed_fragment_reupload_keeps_old_sample(self):
        self.upload_sample("SF", tsv("4\tT\tA\t0/1\n"), fragment=F2.encode())
        self.assertEqual(self.dashboard_map()[(8, "A", "T")], 1)
        # 重传：片段无法定位 -> 拒绝，旧版本保留
        resp = self.upload_sample("SF", tsv("1\tA\tC\t0/1\n"), fragment=b"AAAAAA")
        self.assertIn("上传失败", resp.get_data(as_text=True))
        rows = self.db_query(
            "SELECT s.status, s.frag_start, s.frag_strand, v.pos, v.ref, v.alt "
            "FROM samples s JOIN sample_variants v ON v.sample_id=s.id"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "VALID")
        self.assertEqual((rows[0]["pos"], rows[0]["ref"], rows[0]["alt"]), (8, "A", "T"))
        self.assertEqual(rows[0]["frag_start"], 6)
        self.assertEqual(rows[0]["frag_strand"], "-")

    def test_invalid_fragment_file_rejected(self):
        resp = self.upload_sample("SF", tsv("1\tT\tC\t0/1\n"), fragment=b"ACGN")
        self.assertIn("片段参考只允许包含大写", resp.get_data(as_text=True))
        resp = self.upload_sample("SF", tsv("1\tT\tC\t0/1\n"), fragment=b"")
        self.assertIn("片段参考为空", resp.get_data(as_text=True))
        self.assertEqual(self.db_query("SELECT COUNT(*) c FROM samples")[0]["c"], 0)

    def test_fragment_requires_reference_first(self):
        # 换一个全新的空库环境（尚未上传参考）
        os.environ.pop("DATABASE_PATH", None)
        self.tmp.cleanup()
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.tmp.name, "empty.db")
        app = create_app()
        app.config.update(TESTING=True, SECRET_KEY="test")
        client = app.test_client()
        resp = client.post(
            "/samples",
            data={
                "name": "SF",
                "sample": (io.BytesIO(tsv("1\tT\tC\t0/1\n")), "SF.tsv"),
                "fragment": (io.BytesIO(F1.encode()), "f.txt"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertIn("请先上传参考序列", resp.get_data(as_text=True))


class FragmentRevalidationTest(FragmentAppTestBase):
    def setUp(self):
        super().setUp()
        self.upload_ref(REF_V1)
        # 反向片段样本：SNP + 插入
        self.upload_sample(
            "SF", tsv("POS\tREF\tALT\tGT\n4\tT\tA\t0/1\n2\tC\tCAA\t1/1\n"),
            fragment=F2.encode(),
        )
        # 全长样本：SNP 在 V1/V2/V3 第 2 位均为 C
        self.upload_sample("S1", tsv("2\tC\tA\t0/1\n"))

    def statuses(self):
        return {
            r["name"]: r["status"]
            for r in self.db_query("SELECT name, status FROM samples")
        }

    def test_reference_change_relocates_fragment(self):
        # V4 上 F2 反向唯一匹配 4-9，重验通过并更新定位
        resp = self.upload_ref(REF_V4)
        self.assertIn("参考已更新", resp.get_data(as_text=True))
        self.assertEqual(self.statuses()["SF"], "VALID")
        row = self.db_query(
            "SELECT frag_start, frag_end, frag_strand FROM samples WHERE name='SF'"
        )[0]
        self.assertEqual((row["frag_start"], row["frag_end"], row["frag_strand"]), (4, 9, "-"))
        summary = self.dashboard_map()
        # 换算到新全长坐标：SNP -> (6,A,T)，插入 -> (7,C,CTT)
        self.assertEqual(summary[(6, "A", "T")], 1)
        self.assertEqual(summary[(7, "C", "CTT")], 1)
        # 详情页显示新版本的片段位置
        detail = self.client.get("/variants/6/A>T").get_data(as_text=True)
        self.assertIn("片段 4-9", detail)

    def test_reference_change_fragment_lost_marks_needs_reupload(self):
        # V3 上 F2 正反向均无匹配 -> 待重传，从汇总与详情剔除
        resp = self.upload_ref(REF_V3)
        page = resp.get_data(as_text=True)
        self.assertIn("1 个样本标记为待重传", page)
        self.assertIn("样本「SF」待重传", page)
        self.assertIn("无匹配", page)

        self.assertEqual(self.statuses()["SF"], "NEEDS_REUPLOAD")
        self.assertEqual(self.statuses()["S1"], "VALID")
        # 变体已删除，汇总只剩 S1 的 SNP
        self.assertEqual(
            self.db_query(
                "SELECT COUNT(*) c FROM sample_variants v JOIN samples s "
                "ON s.id=v.sample_id WHERE s.name='SF'"
            )[0]["c"],
            0,
        )
        self.assertEqual(self.dashboard_map(), {(2, "C", "A"): 1})
        # 旧变异详情 404
        self.assertEqual(self.client.get("/variants/8/A>T").status_code, 404)
        # 样本列表显示片段未定位
        index = self.client.get("/").get_data(as_text=True)
        self.assertIn("片段（未定位）", index)

        # 用全长 TSV 重传后恢复
        resp = self.upload_sample("SF", tsv("2\tC\tT\t0/1\n"))
        self.assertIn("成功", resp.get_data(as_text=True))
        self.assertEqual(self.statuses()["SF"], "VALID")
        row = self.db_query(
            "SELECT fragment_seq, frag_start FROM samples WHERE name='SF'"
        )[0]
        self.assertIsNone(row["fragment_seq"])
        self.assertIsNone(row["frag_start"])
        self.assertEqual(self.dashboard_map()[(2, "C", "T")], 1)

    def test_revalidation_uses_retained_fragment_and_tsv(self):
        # 换到 V2：F2 仍唯一匹配 6-11，两行在 V2 下仍合法 -> 自动重验通过
        self.upload_ref(REF_V2)
        self.assertEqual(self.statuses()["SF"], "VALID")
        summary = self.dashboard_map()
        self.assertEqual(summary[(8, "A", "T")], 1)
        self.assertEqual(summary[(9, "C", "CTT")], 1)

    def test_revalidation_row_failure_marks_needs_reupload(self):
        # F4=GTGC 在 V1 上反向唯一匹配 6-9（内部位置，左锚可取）；
        # 在 V5 上反向匹配 1-4（对齐到参考起点），片段末端插入换算时
        # 取不到合法左锚 -> 行级失败，样本待重传。
        f4 = "GTGC"
        ref_v5 = "GCACTTA"  # revcomp(F4)=GCAC 唯一匹配在第 1 位
        self.upload_sample("SG", tsv("4\tC\tCA\t0/1\n"), fragment=f4.encode())
        self.assertEqual(self.statuses()["SG"], "VALID")
        self.assertEqual(self.dashboard_map()[(2, "C", "CT")], 1)

        resp = self.upload_ref(ref_v5)
        self.assertIn("参考已更新", resp.get_data(as_text=True))
        self.assertEqual(self.statuses()["SG"], "NEEDS_REUPLOAD")
        err = self.db_query("SELECT error FROM samples WHERE name='SG'")[0]["error"]
        self.assertIn("第 1 行", err)
        self.assertIn("左锚", err)
        # 失败样本的变异已从汇总剔除
        self.assertNotIn((2, "C", "CT"), self.dashboard_map())


class OldSchemaMigrationTest(unittest.TestCase):
    """既有库（无片段列）升级后照常使用。"""

    OLD_SCHEMA = """
    CREATE TABLE reference_sequences (
        id INTEGER PRIMARY KEY, sequence TEXT NOT NULL, length INTEGER NOT NULL,
        uploaded_at TEXT NOT NULL DEFAULT (datetime('now')));
    CREATE TABLE samples (
        id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
        reference_id INTEGER REFERENCES reference_sequences(id),
        status TEXT NOT NULL CHECK (status IN ('VALID','NEEDS_REUPLOAD')),
        error TEXT, raw_tsv TEXT NOT NULL,
        uploaded_at TEXT NOT NULL DEFAULT (datetime('now')), revalidated_at TEXT);
    CREATE TABLE sample_variants (
        id INTEGER PRIMARY KEY,
        sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
        pos INTEGER NOT NULL, ref TEXT NOT NULL, alt TEXT NOT NULL,
        gt TEXT NOT NULL, kind TEXT NOT NULL,
        orig_pos INTEGER NOT NULL, orig_ref TEXT NOT NULL, orig_alt TEXT NOT NULL);
    """

    def test_migrated_db_keeps_working(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db_path = os.path.join(tmp.name, "old.db")
        conn = sqlite3.connect(db_path)
        conn.executescript(self.OLD_SCHEMA)
        conn.execute(
            "INSERT INTO reference_sequences (sequence, length) VALUES (?, ?)",
            (REF_V1, len(REF_V1)),
        )
        conn.execute(
            "INSERT INTO samples (name, reference_id, status, raw_tsv) "
            "VALUES ('S1', 1, 'VALID', '1\tG\tC\t0/1\n')"
        )
        conn.execute(
            "INSERT INTO sample_variants "
            "(sample_id, pos, ref, alt, gt, kind, orig_pos, orig_ref, orig_alt) "
            "VALUES (1, 1, 'G', 'C', '0/1', 'SNP', 1, 'G', 'C')"
        )
        conn.commit()
        conn.close()

        os.environ["DATABASE_PATH"] = db_path
        self.addCleanup(lambda: os.environ.pop("DATABASE_PATH", None))
        app = create_app()
        app.config.update(TESTING=True, SECRET_KEY="test")

        with app.app_context():
            dash = services.get_dashboard()
            self.assertEqual(dash["stats"]["valid"], 1)
            self.assertEqual(dash["variants"][0]["pos"], 1)
            sample = dash["samples"][0]
            self.assertIsNone(sample["fragment_seq"])
            self.assertIsNone(sample["frag_strand"])

            # 换参考重验在迁移后的库上正常工作
            result = services.upload_reference(REF_V2)
            self.assertEqual(len(result["revalidated"]), 1)
            # 新列已补齐
            from app.db import read_tx
            with read_tx() as conn:
                cols = {
                    r[1] for r in conn.execute("PRAGMA table_info(samples)")
                }
            self.assertIn("fragment_seq", cols)


if __name__ == "__main__":
    unittest.main()
