"""Canonical source ID 解析(按域,带 DOMAIN: 前缀,fail-closed)。

Dataset-specific filename rules:

| 域        | 编辑文件名形态                                          | canonical                        | real 形态                              |
| --------- | ------------------------------------------------------- | -------------------------------- | -------------------------------------- |
| OpenImages| {16hex}_mXX_{8hex}_OpenImages_{model}[_simple].jpg      | OpenImages:{16hex}               | {16hex}.jpg                            |
| CityScapes| {city}_{seq}_{frame}_instance{N}_CityScapes_{model}.jpg | CityScapes:{city}_{seq}_{frame}  | {city}_{seq}_{frame}_leftImg8bit.jpg   |
| SUN_RGBD  | {scene}__{id1}-{id2}_{NNN}_SUN_RGBD_{model}.jpg         | SUN_RGBD:{scene}__{id1}-{id2}(去尾 _NNN) | {scene}__{id1}-{id2}.jpg 或 {NNNNN}.jpg |
| CelebAHQ  | {id}_{attr}_CelebAHQ_{model}.jpg                        | CelebAHQ:{id}                    | {id}.jpg                                |

注意:5 位数字 real(SUN `00000.jpg` 与 CelebAHQ `10010.jpg` 同名形)从文件名无法区分域,
因此 **domain 必须由调用方提供**(labels.json 的 cat 字段为权威);文件名中含域关键字时须与
提供的 domain 一致,否则报错。失败策略:未知格式直接 raise ValueError(列出文件名),
绝不静默退回原始文件名。
"""
from __future__ import annotations

import re

_CITY_EDIT = re.compile(
    r"^(?P<src>[a-z]+(?:-[a-z]+)*_\d+_\d+)(?:_instance\d+)?_CityScapes_(?P<rest>.*)$")
_CITY_REAL = re.compile(
    r"^(?P<src>[a-z]+(?:-[a-z]+)*_\d+_\d+)(?:_leftImg8bit)?\.jpg$")
_OI_EDIT = re.compile(r"^(?P<src>[0-9a-f]{16})(?:_m.*?)?_OpenImages_(?P<rest>.*)$")
_OI_REAL = re.compile(r"^(?P<src>[0-9a-f]{16})\.jpg$")
_SUN_EDIT = re.compile(r"^(?P<stem>.+)_SUN_RGBD_(?P<rest>.*)$")
_SUN_REAL_LONG = re.compile(r"^(?P<stem>.+__\d+(?:-\d+)?)\.jpg$")
_SUN_REAL_NUM = re.compile(r"^(?P<num>\d{5})\.jpg$")
_CELEB_EDIT = re.compile(r"^(?P<src>\d{1,6})_(?P<attr>[a-z_]+)_CelebAHQ_(?P<rest>.*)$")
_CELEB_REAL = re.compile(r"^(?P<src>\d{1,6})\.jpg$")

_DOMAIN_KEYWORDS = ("CityScapes", "OpenImages", "SUN_RGBD", "CelebAHQ")


def _parse_by_domain(name: str, domain: str):
    """按权威 domain 解析 basename → source_id;失败返回 None(由调用方决定抛错)。"""
    if domain == "CityScapes":
        m = _CITY_EDIT.match(name) or _CITY_REAL.match(name)
        return m.group("src") if m else None
    if domain == "OpenImages":
        m = _OI_EDIT.match(name) or _OI_REAL.match(name)
        return m.group("src") if m else None
    if domain == "SUN_RGBD":
        m = _SUN_EDIT.match(name)
        if m:
            stem = m.group("stem")
            cid = "_".join(stem.split("_")[:-1])
            return cid if cid else None
        m = _SUN_REAL_LONG.match(name)
        if m:
            return m.group("stem")
        m = _SUN_REAL_NUM.match(name)  # 5 位数字 real 独立命名空间
        if m:
            return "__real_num__" + m.group("num")
        return None
    if domain == "CelebAHQ":
        m = _CELEB_EDIT.match(name) or _CELEB_REAL.match(name)
        return m.group("src") if m else None
    return None


def canonical_parts(name_or_path: str, domain: str):
    """返回 (domain, source_id)。domain 为权威(来自 labels.cat);
    若文件名含域关键字,须与 domain 一致,否则报错;解析失败抛 ValueError。"""
    name = name_or_path.replace("\\", "/").rsplit("/", 1)[-1]
    if domain not in _DOMAIN_KEYWORDS:
        raise ValueError(f"[canonical_source] 未知 domain: {domain!r} (file: {name})")
    # 文件名含域关键字且与权威 domain 不一致 → 矛盾,报错
    in_name = [d for d in _DOMAIN_KEYWORDS if f"_{d}_" in name or name.startswith(f"{d}_")]
    if in_name and in_name[0] != domain:
        raise ValueError(
            f"[canonical_source] 文件名域关键字 {in_name[0]} 与权威 domain {domain} 矛盾: {name}")
    cid = _parse_by_domain(name, domain)
    if cid is None:
        raise ValueError(f"[canonical_source] 无法解析文件名(domain={domain}): {name}")
    return domain, cid


def canonical_source_id(name_or_path: str, domain: str) -> str:
    """带域前缀的 canonical id,如 'OpenImages:97fef9c4f2e54665'。失败抛 ValueError。"""
    d, cid = canonical_parts(name_or_path, domain)
    return f"{d}:{cid}"
