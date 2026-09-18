"""生成 Excel 版方法×数据集矩阵 (检测/定位两个工作表, 条件格式美化)"""
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import ColorScaleRule

DATASETS = ["INP-X 500", "CelebAHQ", "CocoGlide", "MagicBrush", "SDXL-500"]

det_rows = [
    ("v4 (ours)",   [0.924, 0.808, 0.732, 0.644, 0.658], True),
    ("FLAME",       [0.601, 0.004, 0.035, 0.743, 0.160], False),
    ("MVSS",        [0.464, 0.486, 0.537, 0.534, 0.302], False),
]
det_note = ("检测矩阵 (Image-level detection scores) | AUC where score ranking is available; FLAME CelebAHQ/CocoGlide/SDXL 为固定阈值 recall "
            "(官方输出 AP<0.001, 论文中以 R 标注); INP-X/MagicBrush 为 LAD logit AUC | MVSS: 128px 近似协议")

loc_rows = [
    ("v4 (ours)",    [0.529, 0.293, 0.527, 0.064, 0.076], True),
    ("FLAME-LAD",    [0.128, None,  0.455, 0.259, None],  False),
    ("FLAME full",   [0.140, None,  0.486, None,  None],  False),
    ("ObjectFormer", [0.232, 0.214, 0.402, 0.382, 0.125], False),
    ("MVSS",         [0.113, 0.166, 0.482, 0.384, 0.134], False),
    ("IML-ViT",      [0.145, 0.103, 0.272, 0.379, 0.190], False),
    ("TRAIL",        [0.003, None,  None,  None,  None],  False),
    ("DinoLizer",    [0.260, None,  None,  None,  None],  False),
]
loc_note = ("定位矩阵 (mIoU, 越高越好) | v4: 固定 INP-X 验证阈值 (37x37 grid, global micro); IMDL 方法: INP-X 列为 512px 官方口径逐图 best-thr 均值 "
            "(0.232/0.113/0.145, 与论文主表一致), 其余数据集列为 128px 近似; FLAME: 官方协议; "
            "TRAIL: INP-X mIoU 0.003 (ViT-B/14; CocoGlide patch AUROC 0.736 见 CocoGlide 专门表); DinoLizer 固定 0.5 阈值")

HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=11)
OURS_FILL = PatternFill("solid", fgColor="DDEBF7")
OURS_FONT = Font(bold=True)
NOTE_FONT = Font(italic=True, color="808080", size=9)
BORDER = Border(*[Side(style="thin", color="B0B0B0")] * 4)
CENTER = Alignment(horizontal="center", vertical="center")


def build_sheet(ws, rows, title, note):
    ws.cell(1, 1, title).font = Font(bold=True, size=14)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=6)
    hdr = 3
    for j, name in enumerate(["Method"] + DATASETS):
        c = ws.cell(hdr, j + 1, name)
        c.fill = HEADER_FILL; c.font = HEADER_FONT; c.alignment = CENTER; c.border = BORDER
    for i, (name, vals, is_ours) in enumerate(rows):
        r = hdr + 1 + i
        c = ws.cell(r, 1, name)
        c.font = OURS_FONT if is_ours else Font()
        c.alignment = Alignment(vertical="center")
        c.border = BORDER
        if is_ours:
            c.fill = OURS_FILL
        for j, v in enumerate(vals):
            cc = ws.cell(r, j + 2)
            cc.value = "—" if v is None else round(v, 3)
            cc.alignment = CENTER; cc.border = BORDER
            if is_ours:
                cc.fill = OURS_FILL
    # 色阶 (数值列)
    last_data_row = hdr + len(rows)
    ws.conditional_formatting.add(
        f"C4:F{last_data_row}",
        ColorScaleRule(start_type="num", start_value=0.0, start_color="F8696B",
                       mid_type="num", mid_value=0.5, mid_color="FFEB84",
                       end_type="num", end_value=1.0, end_color="63BE7B"))
    note_r = last_data_row + 2
    ws.cell(note_r, 1, note).font = NOTE_FONT
    ws.merge_cells(start_row=note_r, start_column=1, end_row=note_r, end_column=6)
    ws.column_dimensions["A"].width = 16
    for col in "BCDEF":
        ws.column_dimensions[col].width = 11
    ws.row_dimensions[hdr].height = 22
    ws.freeze_panes = "B4"


wb = Workbook()
ws1 = wb.active
ws1.title = "检测矩阵"
build_sheet(ws1, det_rows, "检测矩阵 (Detection)", det_note)
ws2 = wb.create_sheet("定位矩阵")
build_sheet(ws2, loc_rows, "定位矩阵 (Localization)", loc_note)
wb.save("D:/lunwen/paper/figures/matrices.xlsx")
print("saved matrices.xlsx")
