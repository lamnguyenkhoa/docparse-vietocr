from app.order import reading_order, resolve_overlaps, sort_lines


def reg(name, bbox, kind="text"):
    return {"name": name, "bbox": bbox, "kind": kind}


def names(rs):
    return [r["name"] for r in rs]


def test_two_column_header_left_column_first():
    # Layout gives one block per column (BỘ TÀI CHÍNH / Số ... left, CỘNG HÒA / Độc lập / date right), then the title
    rs = [
        reg("right_block", [1130, 230, 2240, 420]),
        reg("left_block", [400, 225, 1000, 400]),
        reg("title", [820, 480, 1770, 540]),
    ]
    assert names(reading_order(rs)) == ["left_block", "right_block", "title"]


def test_header_as_single_lines_reads_row_by_row():
    # One region per line (form_p1 orphan lines): each row is its own band -> left, right, next row
    rs = [
        reg("cong_hoa", [1130, 230, 2240, 290]),
        reg("ubnd", [510, 230, 930, 290]),
        reg("doc_lap", [1355, 300, 2020, 360]),
        reg("so_noi_vu", [570, 295, 865, 360]),
    ]
    assert names(reading_order(rs)) == ["ubnd", "cong_hoa", "so_noi_vu", "doc_lap"]


def test_touching_boxes_are_not_columns():
    # form_p1: 'Kính gửi:' and 'Họ và tên:' boxes touch by a few px; they must stay in y order (30% band rule)
    rs = [
        reg("ho_va_ten", [350, 605, 590, 662]),
        reg("kinh_gui", [690, 535, 915, 610]),
    ]
    assert names(reading_order(rs)) == ["kinh_gui", "ho_va_ten"]


def test_same_row_fields_left_to_right():
    rs = [reg("gioi_tinh", [1410, 605, 1635, 660]), reg("ho_va_ten", [350, 605, 590, 662])]
    assert names(reading_order(rs)) == ["ho_va_ten", "gioi_tinh"]


def test_resolve_overlaps_drops_nested_same_kind_only():
    big = reg("big", [0, 0, 1000, 1000])
    inner = reg("inner", [100, 100, 200, 200])
    table = reg("table", [100, 300, 400, 400], kind="table")
    assert names(resolve_overlaps([big, inner, table])) == ["big", "table"]


def test_sort_lines_rows_then_x():
    lines = [
        {"bbox": [600, 100, 900, 140]},
        {"bbox": [100, 102, 500, 141]},
        {"bbox": [100, 160, 900, 200]},
    ]
    assert [l["bbox"][0:2] for l in sort_lines(lines)] == [[100, 102], [600, 100], [100, 160]]
