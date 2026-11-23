#!/usr/bin/env python3
"""
YOLO26 辅助标注：对图片做推理，导出 LabelImg 可读标注。

默认格式：Pascal VOC XML（LabelImg 默认）
可选格式：YOLO txt（LabelImg 切换 YOLO 模式时使用）

依赖:
  pip install ultralytics opencv-python pillow

示例:
  python auto_label.py --source ./images --model yolo26n.pt
  python auto_label.py --source ./images --model yolo26s.pt --format yolo --conf 0.35
  python auto_label.py --source ./images --model best.pt --classes person car --save-viz
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO26 推理辅助标注，导出 LabelImg VOC/YOLO 格式"
    )
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="图片文件或目录",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="模型权重，如 yolo26n.pt / yolo26s.pt / 自定义 best.pt",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="标注输出目录，默认与图片同目录（或 source/labels）",
    )
    p.add_argument(
        "--format",
        choices=("voc", "yolo", "both"),
        default="voc",
        help="标注格式：voc=LabelImg Pascal VOC XML，yolo=YOLO txt，both=两种都写",
    )
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    p.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    p.add_argument("--device", type=str, default="", help="如 0 / cpu / '' 自动")
    p.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="只保留这些类别名（不填则保留模型全部检测到的类）",
    )
    p.add_argument(
        "--min-area",
        type=int,
        default=0,
        help="过滤过小框（像素面积），默认不过滤",
    )
    p.add_argument(
        "--save-viz",
        action="store_true",
        help="额外保存画框预览图到 out/viz",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描子目录中的图片",
    )
    return p.parse_args()


def collect_images(source: Path, recursive: bool) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"source 不存在: {source}")
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"不是支持的图片: {source}")
        return [source]

    pattern_iter = source.rglob if recursive else source.glob
    images = [
        p
        for p in pattern_iter("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(images)


def ensure_out_dirs(out_root: Path, fmt: str, save_viz: bool) -> dict[str, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    dirs: dict[str, Path] = {"root": out_root}
    if fmt in ("voc", "both"):
        d = out_root / "Annotations"
        d.mkdir(parents=True, exist_ok=True)
        dirs["voc"] = d
    if fmt in ("yolo", "both"):
        d = out_root / "labels"
        d.mkdir(parents=True, exist_ok=True)
        dirs["yolo"] = d
    if save_viz:
        d = out_root / "viz"
        d.mkdir(parents=True, exist_ok=True)
        dirs["viz"] = d
    return dirs


def clip_box(xyxy, w: int, h: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def write_voc_xml(
    out_path: Path,
    image_path: Path,
    width: int,
    height: int,
    objects: list[dict],
) -> None:
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = image_path.parent.name
    ET.SubElement(ann, "filename").text = image_path.name
    ET.SubElement(ann, "path").text = str(image_path.resolve())

    source = ET.SubElement(ann, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(ann, "segmented").text = "0"

    for obj in objects:
        node = ET.SubElement(ann, "object")
        ET.SubElement(node, "name").text = obj["name"]
        ET.SubElement(node, "pose").text = "Unspecified"
        ET.SubElement(node, "truncated").text = "0"
        ET.SubElement(node, "difficult").text = "0"
        bnd = ET.SubElement(node, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(obj["xmin"])
        ET.SubElement(bnd, "ymin").text = str(obj["ymin"])
        ET.SubElement(bnd, "xmax").text = str(obj["xmax"])
        ET.SubElement(bnd, "ymax").text = str(obj["ymax"])

    rough = ET.tostring(ann, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")
    # minidom 会带 xml 声明；LabelImg 可正常打开
    out_path.write_bytes(pretty)


def write_yolo_txt(out_path: Path, lines: list[str]) -> None:
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_classes_files(out_root: Path, class_names: list[str]) -> None:
    """LabelImg YOLO 模式读 classes.txt；也写一份 predefined_classes.txt 方便导入。"""
    text = "\n".join(class_names) + "\n"
    (out_root / "classes.txt").write_text(text, encoding="utf-8")
    (out_root / "predefined_classes.txt").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "未安装 ultralytics。请先执行:\n"
            "  cd E:\\project\\ultralytics && pip install -e .\n"
            "或: pip install ultralytics",
            file=sys.stderr,
        )
        return 1

    images = collect_images(args.source, args.recursive)
    if not images:
        print(f"未找到图片: {args.source}", file=sys.stderr)
        return 1

    if args.out is None:
        if args.source.is_dir():
            out_root = args.source / "labels_out"
        else:
            out_root = args.source.parent / "labels_out"
    else:
        out_root = args.out

    dirs = ensure_out_dirs(out_root, args.format, args.save_viz)

    print(f"模型: {args.model}")
    print(f"图片数: {len(images)}")
    print(f"输出: {out_root.resolve()}  格式: {args.format}")

    model = YOLO(args.model)
    names: dict[int, str] = dict(model.names)
    allow = set(args.classes) if args.classes else None
    if allow:
        unknown = allow - set(names.values())
        if unknown:
            print(f"警告: 这些类别不在模型中，将被忽略: {sorted(unknown)}")

    # LabelImg 类别表：保持模型原始 id 顺序，便于 YOLO txt 对齐
    ordered_names = [names[i] for i in sorted(names)]
    name_to_id = {n: i for i, n in enumerate(ordered_names)}
    write_classes_files(out_root, ordered_names)

    predict_kw = dict(
        source=[str(p) for p in images],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        stream=True,
        verbose=False,
        save=False,
    )
    if args.device:
        predict_kw["device"] = args.device

    total_boxes = 0
    done = 0
    for result in model.predict(**predict_kw):
        done += 1
        image_path = Path(result.path)
        h, w = result.orig_shape  # H, W

        objects_voc: list[dict] = []
        yolo_lines: list[str] = []

        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for box, cid, conf in zip(xyxy, cls_ids, confs):
                cname = names.get(int(cid), str(cid))
                if allow is not None and cname not in allow:
                    continue
                clipped = clip_box(box, w, h)
                if clipped is None:
                    continue
                x1, y1, x2, y2 = clipped
                area = (x2 - x1) * (y2 - y1)
                if area < args.min_area:
                    continue

                objects_voc.append(
                    {
                        "name": cname,
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2,
                        "conf": float(conf),
                    }
                )

                # YOLO: class_id xc yc w h （归一化，相对整图）
                xc = ((x1 + x2) / 2.0) / w
                yc = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                yolo_id = name_to_id[cname]
                yolo_lines.append(
                    f"{yolo_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                )

        total_boxes += len(objects_voc)

        if args.format in ("voc", "both"):
            xml_path = dirs["voc"] / f"{image_path.stem}.xml"
            write_voc_xml(xml_path, image_path, w, h, objects_voc)

        if args.format in ("yolo", "both"):
            txt_path = dirs["yolo"] / f"{image_path.stem}.txt"
            write_yolo_txt(txt_path, yolo_lines)

        if args.save_viz:
            plotted = result.plot()  # BGR ndarray
            import cv2

            viz_path = dirs["viz"] / image_path.name
            cv2.imwrite(str(viz_path), plotted)

        if done % 20 == 0 or done == len(images):
            print(f"[{done}/{len(images)}] {image_path.name} -> {len(objects_voc)} boxes")

    # LabelImg 使用 VOC 时，图片通常与 Annotations 同级的 JPEGImages
    # 写一份简短说明到输出目录
    tip = out_root / "README_labelimg.txt"
    tip.write_text(
        "\n".join(
            [
                "LabelImg 打开方式:",
                "1) Pascal VOC: Open Dir 指向图片目录；Change Save Dir 指向 Annotations/",
                "2) YOLO: 先 View -> Auto Save；格式选 YOLO；Open Dir 图片目录；",
                "   Change Save Dir 指向 labels/；用 Open Dir 时会读取同级 classes.txt",
                "   也可把 predefined_classes.txt 内容导入 LabelImg 的预定义类别。",
                "",
                f"本次格式: {args.format}",
                f"类别文件: {out_root / 'classes.txt'}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"完成: {done} 张图, 共 {total_boxes} 个框")
    print(f"输出目录: {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
