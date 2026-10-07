#!/usr/bin/env python3
"""Print the op tree of a compiled ``.mpsgraph`` (MLIR bytecode, ``mps`` dialect).

Core AI's MLIR bindings cannot parse these files: the ``mps`` dialect is not
registered, so its custom attributes and types cannot be decoded. The IR
structure is generic MLIR bytecode, though, so this reader walks it without
the dialect: op names, nesting, operands, result types (builtin types are
decoded; others print as ``!attrN`` / ``!typeN`` or their stored asm text) and
symbol names. Enough to see what a ``GPU_region`` holds.

Usage::

    python scripts/dump_mpsgraph.py specialized_model_0.mpsgraph [--func NAME]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bytecode section ids.
S_STRING, S_DIALECT, S_ATTRTYPE, S_ATTRTYPE_OFFSET, S_IR = 0, 1, 2, 3, 4
S_RESOURCE, S_RESOURCE_OFFSET, S_DIALECT_VERSIONS, S_PROPERTIES = 5, 6, 7, 8

# Op encoding mask bits.
M_ATTRS, M_RESULTS, M_OPERANDS, M_SUCCESSORS = 0x01, 0x02, 0x04, 0x08
M_REGIONS, M_USELIST, M_PROPERTIES = 0x10, 0x20, 0x40

# Bytecode versions with format changes.
V_DIALECT_VERSIONING, V_LAZY_LOADING, V_USELIST = 1, 2, 3
V_ELIDE_ARG_LOC, V_NATIVE_PROPERTIES = 4, 5


class Reader:
    def __init__(self, data: bytes, pos: int = 0, end: int | None = None) -> None:
        self.data = data
        self.pos = pos
        self.end = len(data) if end is None else end

    def done(self) -> bool:
        return self.pos >= self.end

    def byte(self) -> int:
        b = self.data[self.pos]
        self.pos += 1
        return b

    def bytes(self, n: int) -> bytes:
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def varint(self) -> int:
        b0 = self.data[self.pos]
        if b0 & 1:
            self.pos += 1
            return b0 >> 1
        if b0 == 0:
            v = int.from_bytes(self.data[self.pos + 1 : self.pos + 9], "little")
            self.pos += 9
            return v
        n = (b0 & -b0).bit_length()  # trailing zeros + 1
        v = int.from_bytes(self.data[self.pos : self.pos + n], "little") >> n
        self.pos += n
        return v

    def svarint(self) -> int:
        v = self.varint()
        return (v >> 1) ^ -(v & 1)

    def flag_varint(self) -> tuple[int, bool]:
        v = self.varint()
        return v >> 1, bool(v & 1)

    def cstr(self) -> str:
        end = self.data.index(b"\0", self.pos)
        s = self.data[self.pos : end].decode("utf-8", "replace")
        self.pos = end + 1
        return s

    def section(self) -> tuple[int, int, int]:
        """Return (id, data_start, data_end) and move past the section."""
        head = self.byte()
        sid, aligned = head & 0x7F, head & 0x80
        length = self.varint()
        if aligned:
            align = self.varint()
            while self.pos % align:
                if self.byte() != 0xCB:
                    raise ValueError("bad alignment padding")
        start = self.pos
        self.pos += length
        return sid, start, start + length


class Bytecode:
    def __init__(self, data: bytes) -> None:
        r = Reader(data)
        if r.bytes(4) != b"ML\xefR":
            raise ValueError("not MLIR bytecode")
        self.version = r.varint()
        self.producer = r.cstr()
        self.data = data
        self.sections: dict[int, tuple[int, int]] = {}
        while not r.done():
            sid, start, end = r.section()
            self.sections[sid] = (start, end)
        self._strings()
        self._dialects()
        self._attr_type_offsets()
        self.attr_cache: dict[int, str] = {}
        self.type_cache: dict[int, str] = {}

    def _sec(self, sid: int) -> Reader:
        start, end = self.sections[sid]
        return Reader(self.data, start, end)

    def _strings(self) -> None:
        r = self._sec(S_STRING)
        n = r.varint()
        sizes = [r.varint() for _ in range(n)][::-1]
        self.strings = []
        for size in sizes:
            self.strings.append(r.bytes(size)[:-1].decode("utf-8", "replace"))

    def _dialects(self) -> None:
        r = self._sec(S_DIALECT)
        n = r.varint()
        self.dialects = []
        for _ in range(n):
            if self.version >= V_DIALECT_VERSIONING:
                idx, has_version = r.flag_varint()
                if has_version:
                    r.section()  # kDialectVersions blob
            else:
                idx = r.varint()
            self.dialects.append(self.strings[idx])
        self.op_names = []
        total = r.varint() if self.version >= V_NATIVE_PROPERTIES else None
        while not r.done() and (total is None or len(self.op_names) < total):
            dialect = self.dialects[r.varint()]
            for _ in range(r.varint()):
                if self.version >= V_NATIVE_PROPERTIES:
                    idx, _registered = r.flag_varint()
                else:
                    idx = r.varint()
                self.op_names.append(f"{dialect}.{self.strings[idx]}")

    def _attr_type_offsets(self) -> None:
        r = self._sec(S_ATTRTYPE_OFFSET)
        num_attrs, num_types = r.varint(), r.varint()
        base = self.sections[S_ATTRTYPE][0]
        offset = 0

        def read(count: int) -> list[tuple[str, int, int, bool]]:
            nonlocal offset
            out: list[tuple[str, int, int, bool]] = []
            while len(out) < count:
                dialect = self.dialects[r.varint()]
                for _ in range(r.varint()):
                    size, custom = r.flag_varint()
                    out.append((dialect, base + offset, base + offset + size, custom))
                    offset += size
            return out

        self.attrs = read(num_attrs)
        self.types = read(num_types)

    # -- attributes / types -------------------------------------------------

    def attr(self, idx: int) -> str:
        if idx not in self.attr_cache:
            self.attr_cache[idx] = f"#attr{idx}"  # cycle guard
            self.attr_cache[idx] = self._attr(idx)
        return self.attr_cache[idx]

    def type(self, idx: int) -> str:
        if idx not in self.type_cache:
            self.type_cache[idx] = f"!type{idx}"
            self.type_cache[idx] = self._type(idx)
        return self.type_cache[idx]

    def _attr(self, idx: int) -> str:
        dialect, start, end, custom = self.attrs[idx]
        r = Reader(self.data, start, end)
        if not custom:
            return r.cstr()
        if dialect != "builtin":
            return f"#{dialect}<attr{idx}>"
        code = r.varint()
        try:
            if code == 0:  # ArrayAttr
                return "[" + ", ".join(self.attr(r.varint()) for _ in range(r.varint())) + "]"
            if code == 1:  # DictionaryAttr
                items = []
                for _ in range(r.varint()):
                    k, v = r.varint(), r.varint()
                    items.append(f"{self.attr(k).strip(chr(34))} = {self.attr(v)}")
                return "{" + ", ".join(items) + "}"
            if code == 2:  # StringAttr
                return '"' + self.strings[r.varint()] + '"'
            if code == 3:  # StringAttr with type
                return '"' + self.strings[r.varint()] + '"'
            if code == 4:  # FlatSymbolRefAttr
                return "@" + self.attr(r.varint()).strip('"')
            if code == 5:  # SymbolRefAttr
                root = self.attr(r.varint()).strip('"')
                nested = [self.attr(r.varint()) for _ in range(r.varint())]
                return "@" + "::".join([root, *nested])
            if code == 6:  # TypeAttr
                return self.type(r.varint())
            if code == 7:
                return "unit"
            if code == 8:  # IntegerAttr
                ty = self.type(r.varint())
                if ty.startswith(("i", "si", "ui", "index")) and ty != "index" and int(ty.lstrip("sui")) <= 64:
                    return f"{r.svarint()} : {ty}"
                return f"int : {ty}"
            if code == 11:  # FileLineColLoc
                return f"loc({self.attr(r.varint())}:{r.varint()}:{r.varint()})"
            if code == 12:  # FusedLoc
                return "loc(fused" + self.attr_list(r) + ")"
            if code == 13:  # FusedLoc with metadata
                locs = self.attr_list(r)
                return f"loc(fused<{self.attr(r.varint())}>{locs})"
            if code == 14:  # NameLoc
                return f"loc({self.attr(r.varint())} at {self.attr(r.varint())})"
            if code == 15:
                return "loc(unknown)"
        except (IndexError, ValueError):
            pass
        return f"#builtin<code{code}>"

    def attr_list(self, r: Reader) -> str:
        return "[" + ", ".join(self.attr(r.varint()) for _ in range(r.varint())) + "]"

    def _type(self, idx: int) -> str:
        dialect, start, end, custom = self.types[idx]
        r = Reader(self.data, start, end)
        if not custom:
            return r.cstr()
        if dialect != "builtin":
            return f"!{dialect}<type{idx}>"
        code = r.varint()
        simple = {1: "index", 3: "bf16", 4: "f16", 5: "f32", 6: "f64", 7: "f80", 8: "f128", 12: "none"}
        if code in simple:
            return simple[code]
        if code == 0:  # IntegerType
            v = r.varint()
            width, sign = v >> 2, v & 3
            return ("i", "si", "ui")[sign] + str(width) if sign < 3 else f"i{width}"
        if code == 2:  # FunctionType
            ins = [self.type(r.varint()) for _ in range(r.varint())]
            outs = [self.type(r.varint()) for _ in range(r.varint())]
            return f"({', '.join(ins)}) -> ({', '.join(outs)})"
        if code in (13, 14):  # RankedTensorType [with encoding]
            enc = self.attr(r.varint()) if code == 14 else None
            dims = ["?" if d == -(1 << 63) else str(d) for d in (r.svarint() for _ in range(r.varint()))]
            elt = self.type(r.varint())
            body = "x".join([*dims, elt])
            return f"tensor<{body}{', ' + enc if enc else ''}>"
        if code == 18:  # UnrankedTensorType
            return f"tensor<*x{self.type(r.varint())}>"
        if code in (10, 11):  # MemRefType [with memory space]: shape, element type, layout
            space = self.attr(r.varint()) if code == 11 else None
            dims = ["?" if d == -(1 << 63) else str(d) for d in (r.svarint() for _ in range(r.varint()))]
            elt = self.type(r.varint())
            return f"memref<{'x'.join([*dims, elt])}{', ' + space if space else ''}>"
        return f"!builtin<code{code}>"

    # -- IR -----------------------------------------------------------------

    def walk(self, emit) -> None:
        start, end = self.sections[S_IR]
        r = Reader(self.data, start, end)
        # One value table per isolated scope; each open region owns a reserved
        # slice of it (MLIR numbers a region's values before nested regions').
        self.scopes: list[tuple[list[str | None], list[int]]] = [([], [0])]
        self.counter = 0
        self._block(r, 0, emit, top=True)

    def _new_value(self, ty: str) -> str:
        name = f"%{self.counter}"
        self.counter += 1
        values, next_ids = self.scopes[-1]
        if next_ids[-1] >= len(values):
            values.append(None)
        values[next_ids[-1]] = name
        next_ids[-1] += 1
        self.value_types[name] = ty
        return name

    def _block(self, r: Reader, depth: int, emit, top: bool = False) -> None:
        num_ops, has_args = r.flag_varint()
        args = []
        if has_args:
            for _ in range(r.varint()):
                if self.version >= V_ELIDE_ARG_LOC:
                    ty, has_loc = r.flag_varint()
                    if has_loc:
                        r.varint()
                else:
                    ty = r.varint()
                    r.varint()
                args.append(self._new_value(self.type(ty)))
            if self.version >= V_USELIST:
                if r.byte() & M_USELIST:
                    self._skip_use_lists(r, len(args) > 1)
        if args:
            emit(depth, "^bb(" + ", ".join(f"{a}: {self.value_types[a]}" for a in args) + ")")
        for _ in range(num_ops):
            self._op(r, depth, emit)

    @staticmethod
    def _skip_use_lists(r: Reader, multiple: bool) -> None:
        # A single value carries one order with no count and no value index.
        for _ in range(r.varint() if multiple else 1):
            if multiple:
                r.varint()  # value index
            count, _pairs = r.flag_varint()
            for _ in range(count):
                r.varint()

    def _region(self, r: Reader, depth: int, emit, isolated: bool) -> None:
        num_blocks = r.varint()
        if not num_blocks:
            return
        num_values = r.varint()
        if isolated:
            self.scopes.append(([], []))
        values, next_ids = self.scopes[-1]
        mark = len(values)
        next_ids.append(mark)
        values.extend([None] * num_values)
        for _ in range(num_blocks):
            self._block(r, depth, emit)
        next_ids.pop()
        if isolated:
            self.scopes.pop()
        else:
            del values[mark:]

    def _op(self, r: Reader, depth: int, emit) -> None:
        name = self.op_names[r.varint()]
        mask = r.byte()
        loc = r.varint()
        attrs = self.attr(r.varint()) if mask & M_ATTRS else ""
        props = r.varint() if mask & M_PROPERTIES else None
        result_types = [self.type(r.varint()) for _ in range(r.varint())] if mask & M_RESULTS else []
        operands = []
        if mask & M_OPERANDS:
            for _ in range(r.varint()):
                idx = r.varint()
                values = self.scopes[-1][0]
                operands.append((values[idx] if idx < len(values) else None) or f"%?{idx}")
        if mask & M_SUCCESSORS:
            for _ in range(r.varint()):
                r.varint()
        if mask & M_USELIST:
            self._skip_use_lists(r, len(result_types) > 1)
        results = [self._new_value(t) for t in result_types]
        line = ""
        if results:
            line += ", ".join(results) + " = "
        line += name
        if operands:
            line += "(" + ", ".join(operands) + ")"
        if attrs:
            line += " " + attrs
        if props is not None:
            line += f" <props{props}>"
        if result_types:
            line += " : " + ", ".join(result_types)
        emit(depth, line, loc=loc)
        if mask & M_REGIONS:
            num_regions, isolated = r.flag_varint()
            for _ in range(num_regions):
                emit(depth, "{")
                if isolated and self.version >= V_LAZY_LOADING:
                    sid, start, end = r.section()
                    if sid != S_IR:
                        raise ValueError(f"expected IR section, got {sid}")
                    self._region(Reader(self.data, start, end), depth + 1, emit, isolated=True)
                else:
                    self._region(r, depth + 1, emit, isolated=isolated)
                emit(depth, "}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph", type=Path, help=".mpsgraph file (MLIR bytecode)")
    ap.add_argument("--func", help="only print ops under the op whose sym_name contains this")
    ap.add_argument("--locs", action="store_true", help="append each op's location")
    args = ap.parse_args(argv)

    bc = Bytecode(args.graph.read_bytes())
    bc.value_types = {}
    print(f"// bytecode v{bc.version} producer={bc.producer} dialects={bc.dialects}")
    keep_depth: list[int | None] = [None]

    def emit(depth: int, line: str, loc: int | None = None) -> None:
        if args.func:
            if keep_depth[0] is not None and depth <= keep_depth[0] and line not in ("{", "}"):
                keep_depth[0] = None
            if keep_depth[0] is None:
                if f'sym_name = "{args.func}' in line or (f'"{args.func}' in line and "sym_name" in line):
                    keep_depth[0] = depth
                else:
                    return
        if args.locs and loc is not None:
            line += f"  // {bc.attr(loc)}"
        print("  " * depth + line)

    bc.walk(emit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
