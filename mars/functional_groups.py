"""
AccFG-style functional group identification using graph-based topology.

Identifies functional groups at the molecule level by walking the molecular
graph (neighbors, bond_orders) produced by ``_build_topology`` in ``ir.py``.
Uses hierarchical deduplication so that more-specific groups suppress their
generic parents (e.g. "ester" suppresses standalone "C=O" and "C-O" on the
same atoms).

No external dependencies beyond NumPy.
"""

from typing import Dict, List, Optional, Set, Tuple

# Bond order symbol map (mirrors ir.py)
_BO_SYM = {1: "-", 2: "=", 3: "#"}


# ============================================================================
# Graph helpers — thin wrappers around the topology dict
# ============================================================================


class MolGraph:
    """Lightweight read-only molecular graph backed by a topology dict."""

    __slots__ = ("symbols", "nbrs", "bo_map", "_n")

    def __init__(self, symbols: list, topology: dict):
        self.symbols = symbols
        self.nbrs: Dict[int, Set[int]] = topology.get("neighbors", {})
        self.bo_map: Dict[Tuple[int, int], int] = topology.get("bond_orders", {})
        self._n = len(symbols)

    # -- atom queries ---------------------------------------------------------
    def sym(self, i: int) -> str:
        return self.symbols[i] if 0 <= i < self._n else "?"

    def bo(self, i: int, j: int) -> int:
        return self.bo_map.get((min(i, j), max(i, j)), 1)

    def neighbors(self, i: int) -> Set[int]:
        return self.nbrs.get(i, set())

    def heavy_neighbors(self, i: int) -> List[int]:
        return [n for n in self.neighbors(i) if self.sym(n) != "H"]

    def h_count(self, i: int) -> int:
        return sum(1 for n in self.neighbors(i) if self.sym(n) == "H")

    def has_bond_to(self, i: int, element: str, order: int) -> bool:
        return any(self.bo(i, n) == order and self.sym(n) == element for n in self.neighbors(i))

    def neighbors_with(self, i: int, element: str, order: int) -> List[int]:
        return [n for n in self.neighbors(i) if self.sym(n) == element and self.bo(i, n) == order]

    # -- ring queries ---------------------------------------------------------
    def is_in_ring(self, i: int) -> bool:
        nl = list(self.neighbors(i))
        for n in nl:
            if self.neighbors(n) & set(nl) - {i}:
                return True
        return False

    def is_in_3ring(self, i: int) -> bool:
        nl = list(self.neighbors(i))
        for a in range(len(nl)):
            for b in range(a + 1, len(nl)):
                if nl[b] in self.neighbors(nl[a]):
                    return True
        return False

    def is_sp2_ring(self, i: int) -> bool:
        if not self.is_in_ring(i):
            return False
        return any(self.bo(i, n) == 2 for n in self.neighbors(i) if self.is_in_ring(n))

    def is_aromatic_C(self, i: int) -> bool:
        return self.sym(i) == "C" and self.is_sp2_ring(i)

    def is_pyrrole_N(self, i: int) -> bool:
        return self.sym(i) == "N" and self.is_sp2_ring(i) and self.h_count(i) >= 1

    def has_heteroatom_ring_nbr(self, i: int) -> bool:
        return any(self.sym(n) in ("N", "O", "S") for n in self.neighbors(i) if self.is_in_ring(n))

    @property
    def n_atoms(self) -> int:
        return self._n

    def atoms_of(self, element: str):
        """Iterate over indices of atoms with a given element symbol."""
        for i in range(self._n):
            if self.symbols[i] == element:
                yield i


# ============================================================================
# Functional group pattern matchers
# ============================================================================
# Each returns a list of (name, atom_tuple) pairs.
# atom_tuple contains the indices of ALL atoms in the FG instance.


def _match_carbonyl_family(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    """Identify C=O groups and classify by context."""
    results = []
    for c in g.atoms_of("C"):
        o_list = g.neighbors_with(c, "O", 2)
        if not o_list:
            continue
        o = o_list[0]
        hn_c = g.heavy_neighbors(c)
        hn_c_no_o = [n for n in hn_c if n != o]

        o_single = [n for n in hn_c if g.sym(n) == "O" and g.bo(c, n) == 1]
        n_adj = [n for n in hn_c if g.sym(n) == "N"]
        hal_adj = [n for n in hn_c if g.sym(n) in ("F", "Cl", "Br", "I")]
        s_adj = [n for n in hn_c if g.sym(n) == "S"]

        if g.h_count(c) >= 1:
            # Aldehyde: C(=O)H
            h_atoms = tuple(n for n in g.neighbors(c) if g.sym(n) == "H")
            results.append(("aldehyde", (c, o) + h_atoms))
            continue

        if o_single:
            o2 = o_single[0]
            # Check for anhydride: C(=O)-O-C(=O)
            if any(g.has_bond_to(n2, "O", 2) for n2 in g.heavy_neighbors(o2) if n2 != c):
                results.append(("anhydride", (c, o, o2)))
                continue
            # Carboxylic acid: C(=O)(OH)
            if g.h_count(o2) >= 1:
                h_on_o = tuple(n for n in g.neighbors(o2) if g.sym(n) == "H")
                results.append(("carboxylic_acid", (c, o, o2) + h_on_o))
                continue
            # Carbonate: C(=O)(O-)(O-)
            if len(o_single) >= 2:
                results.append(("carbonate", (c, o, o_single[0], o_single[1])))
                continue
            # Carbamate: C(=O)(O)(N)
            if n_adj:
                results.append(("carbamate", (c, o, o2, n_adj[0])))
                continue
            # Lactone vs Ester
            if g.is_in_ring(c):
                results.append(("lactone", (c, o, o2)))
            else:
                results.append(("ester", (c, o, o2)))
            continue

        if hal_adj:
            results.append(("acyl_halide", (c, o, hal_adj[0])))
            continue

        if n_adj:
            if len(n_adj) >= 2:
                results.append(("urea", (c, o, n_adj[0], n_adj[1])))
            elif g.is_in_ring(c):
                results.append(("lactam", (c, o, n_adj[0])))
            else:
                results.append(("amide", (c, o, n_adj[0])))
            continue

        if s_adj:
            results.append(("thioester", (c, o, s_adj[0])))
            continue

        # Default: ketone
        results.append(("ketone", (c, o)))

    return results


def _match_thiocarbonyl(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    results = []
    for c in g.atoms_of("C"):
        s_list = g.neighbors_with(c, "S", 2)
        if not s_list:
            continue
        s = s_list[0]
        if any(g.sym(n) == "N" for n in g.heavy_neighbors(c) if n != s):
            results.append(("thioamide", (c, s)))
        else:
            results.append(("thioketone", (c, s)))
    return results


def _match_cc_double(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    results = []
    seen = set()
    for c1 in g.atoms_of("C"):
        for c2 in g.neighbors_with(c1, "C", 2):
            pair = (min(c1, c2), max(c1, c2))
            if pair in seen:
                continue
            seen.add(pair)

            if g.is_aromatic_C(c1) or g.is_aromatic_C(c2):
                ring_adj = {
                    n for n in list(g.neighbors(c1)) + list(g.neighbors(c2)) if g.is_in_ring(n)
                }
                if any(g.sym(n) in ("N", "O", "S") for n in ring_adj):
                    results.append(("heteroaromatic_ring", pair))
                else:
                    results.append(("aromatic_ring", pair))
                continue

            # Allene / cumulene
            if (
                g.has_bond_to(c1, "C", 2)
                and g.has_bond_to(c2, "C", 2)
                and len(g.neighbors_with(c1, "C", 2)) >= 2
            ):
                results.append(("allene", pair))
                continue

            hn_both = g.heavy_neighbors(c1) + g.heavy_neighbors(c2)
            if any(g.sym(n) == "N" for n in hn_both if n not in pair):
                results.append(("enamine", pair))
            elif any(g.sym(n) == "O" for n in hn_both if n not in pair):
                results.append(("enol_ether", pair))
            else:
                results.append(("alkene", pair))
    return results


def _match_cn_double(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    results = []
    for c in g.atoms_of("C"):
        for n in g.neighbors_with(c, "N", 2):
            # Check oxime: N-OH
            if any(g.sym(nb) == "O" and g.h_count(nb) >= 1 for nb in g.neighbors(n)):
                results.append(("oxime", (c, n)))
            elif any(g.sym(nb) == "N" for nb in g.heavy_neighbors(n)):
                results.append(("hydrazone", (c, n)))
            elif g.is_in_ring(n):
                results.append(("imine_ring", (c, n)))
            else:
                results.append(("imine", (c, n)))
    return results


def _match_triple_bonds(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    results = []
    seen = set()
    for c in g.atoms_of("C"):
        for n in g.neighbors_with(c, "N", 3):
            pair = (min(c, n), max(c, n))
            if pair not in seen:
                seen.add(pair)
                results.append(("nitrile", pair))
        for c2 in g.neighbors_with(c, "C", 3):
            pair = (min(c, c2), max(c, c2))
            if pair not in seen:
                seen.add(pair)
                results.append(("alkyne", pair))
    return results


def _match_nn_bonds(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    results = []
    seen = set()
    for n1 in g.atoms_of("N"):
        for n2 in g.neighbors(n1):
            if g.sym(n2) != "N":
                continue
            pair = (min(n1, n2), max(n1, n2))
            if pair in seen:
                continue
            seen.add(pair)
            order = g.bo(n1, n2)
            if order == 3:
                results.append(("diazonium", pair))
            elif order == 2:
                results.append(("azo", pair))
            else:
                results.append(("hydrazine", pair))
    return results


def _match_no_bonds(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    results = []
    for n_idx in g.atoms_of("N"):
        o_nbrs = [nb for nb in g.neighbors(n_idx) if g.sym(nb) == "O"]
        if not o_nbrs:
            continue
        o_double = [ob for ob in o_nbrs if g.bo(n_idx, ob) == 2]
        o_single = [ob for ob in o_nbrs if g.bo(n_idx, ob) == 1]

        if len(o_double) >= 1 and len(o_nbrs) >= 2:
            # Nitro: N(=O)(=O) or N(=O)(O-)
            atoms = (n_idx,) + tuple(o_nbrs)
            results.append(("nitro", atoms))
        elif len(o_double) == 1 and len(o_nbrs) == 1:
            results.append(("nitroso", (n_idx, o_double[0])))
        elif o_single and any(g.has_bond_to(n_idx, "O", 2) for _ in [1]):
            # N-oxide: N(=O)(single O already counted)
            pass  # handled above
        elif o_single:
            # N-O single bond: N-oxide or hydroxylamine
            for o in o_single:
                if any(g.has_bond_to(n_idx, el, 2) for el in ("O", "C")):
                    results.append(("n_oxide", (n_idx, o)))
                else:
                    results.append(("hydroxylamine", (n_idx, o)))
    return results


def _match_oo_bonds(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    results = []
    seen = set()
    for o1 in g.atoms_of("O"):
        for o2 in g.neighbors(o1):
            if g.sym(o2) != "O":
                continue
            pair = (min(o1, o2), max(o1, o2))
            if pair in seen:
                continue
            seen.add(pair)
            if g.h_count(o1) + g.h_count(o2) >= 1:
                results.append(("hydroperoxide", pair))
            else:
                results.append(("peroxide", pair))
    return results


def _match_ss_bonds(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    results = []
    seen = set()
    for s1 in g.atoms_of("S"):
        for s2 in g.neighbors(s1):
            if g.sym(s2) != "S":
                continue
            pair = (min(s1, s2), max(s1, s2))
            if pair not in seen:
                seen.add(pair)
                results.append(("disulfide", pair))
    return results


def _match_so_bonds(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    results = []
    for s in g.atoms_of("S"):
        o_double = g.neighbors_with(s, "O", 2)
        if not o_double:
            continue
        n_so2 = len(o_double)
        atoms = (s,) + tuple(o_double)
        if n_so2 >= 2:
            hn_s = g.heavy_neighbors(s)
            if any(g.sym(n) == "O" and g.h_count(n) >= 1 for n in hn_s):
                results.append(("sulfonic_acid", atoms))
            elif any(g.sym(n) == "N" for n in hn_s):
                results.append(("sulfonamide", atoms))
            else:
                results.append(("sulfone", atoms))
        else:
            results.append(("sulfoxide", atoms))
    return results


def _match_po_bonds(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    results = []
    for p in g.atoms_of("P"):
        o_double = g.neighbors_with(p, "O", 2)
        o_single = g.neighbors_with(p, "O", 1)
        if o_double:
            o = o_double[0]
            hn_p = g.heavy_neighbors(p)
            if any(g.sym(n) == "N" for n in hn_p):
                results.append(("phosphonamide", (p, o)))
            elif any(g.sym(n) == "O" and g.h_count(n) >= 1 for n in hn_p):
                results.append(("phosphonic_acid", (p, o)))
            else:
                results.append(("phosphate", (p, o)))
        for o in o_single:
            results.append(("phosphate_ester", (p, o)))
    return results


def _match_alcohol_ether(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    """Match O with single bonds: alcohol, phenol, ether, epoxide, vinyl ether."""
    results = []
    for o in g.atoms_of("O"):
        # Skip O already in double bonds (carbonyl, S=O, etc.)
        if any(g.bo(o, n) >= 2 for n in g.neighbors(o)):
            continue
        hn_o = g.heavy_neighbors(o)
        if not hn_o:
            continue

        # O-H present?
        has_h = g.h_count(o) >= 1

        if has_h and len(hn_o) == 1:
            c = hn_o[0]
            if g.sym(c) == "C":
                # Skip if this O is part of carboxylic acid (C=O neighbor)
                if g.has_bond_to(c, "O", 2):
                    continue
                if g.is_aromatic_C(c):
                    results.append(("phenol", (o, c)))
                elif g.has_bond_to(c, "C", 2):
                    results.append(("enol", (o, c)))
                else:
                    results.append(("alcohol", (o, c)))
            continue

        if len(hn_o) == 2 and not has_h:
            c1, c2 = hn_o[0], hn_o[1]
            # O-O already handled
            if g.sym(c1) == "O" or g.sym(c2) == "O":
                continue
            # Skip if part of ester/anhydride (C=O neighbor)
            if any(g.has_bond_to(c, "O", 2) for c in (c1, c2) if g.sym(c) == "C"):
                continue
            if g.is_in_3ring(o):
                results.append(("epoxide", (o, c1, c2)))
            elif any(g.has_bond_to(c, "C", 2) for c in (c1, c2) if g.sym(c) == "C"):
                results.append(("vinyl_ether", (o, c1, c2)))
            else:
                results.append(("ether", (o, c1, c2)))
    return results


def _match_amine(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    """Match N-based groups: amines, aniline, guanidine, amidine."""
    results = []
    for n in g.atoms_of("N"):
        # Skip N already in double/triple bonds or amide/nitro/etc.
        if any(g.bo(n, nb) >= 2 for nb in g.neighbors(n)):
            continue
        # Skip if N bonded to C(=O) — that's amide, handled in carbonyl family
        if any(g.sym(nb) == "C" and g.has_bond_to(nb, "O", 2) for nb in g.heavy_neighbors(n)):
            continue
        # Skip if N bonded to S(=O) — sulfonamide
        if any(g.sym(nb) == "S" and g.has_bond_to(nb, "O", 2) for nb in g.heavy_neighbors(n)):
            continue

        hn_n = g.heavy_neighbors(n)
        hc_n = g.h_count(n)
        h_atoms = tuple(nb for nb in g.neighbors(n) if g.sym(nb) == "H")

        # Pyrrole N
        if g.is_pyrrole_N(n):
            results.append(("pyrrole_n", (n,) + h_atoms))
            continue

        # Hydrazine (N-N)
        if any(g.sym(nb) == "N" for nb in hn_n):
            continue  # handled in _match_nn_bonds

        # Guanidine/amidine: N bonded to C that has >=2 N neighbors
        c_nbrs = [nb for nb in hn_n if g.sym(nb) == "C"]
        if c_nbrs:
            for c in c_nbrs:
                n_on_c = sum(1 for nb2 in g.heavy_neighbors(c) if g.sym(nb2) == "N")
                if n_on_c >= 2 and not g.has_bond_to(c, "O", 2):
                    results.append(("guanidine", (n, c) + h_atoms))
                    break
            else:
                # Regular amine
                if any(g.is_aromatic_C(nb) for nb in hn_n):
                    results.append(("aniline", (n,) + tuple(hn_n) + h_atoms))
                elif hc_n >= 2:
                    results.append(("primary_amine", (n,) + h_atoms))
                elif hc_n == 1:
                    results.append(("secondary_amine", (n,) + h_atoms))
                else:
                    results.append(("tertiary_amine", (n,) + tuple(hn_n)))
        else:
            if hc_n >= 2:
                results.append(("primary_amine", (n,) + h_atoms))
            elif hc_n == 1:
                results.append(("secondary_amine", (n,) + h_atoms))
            else:
                results.append(("tertiary_amine", (n,) + tuple(hn_n)))
    return results


def _match_thiol_thioether(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    results = []
    for s in g.atoms_of("S"):
        # Skip S in double bonds (sulfoxide, thiocarbonyl, etc.)
        if any(g.bo(s, n) >= 2 for n in g.neighbors(s)):
            continue
        # Skip S-S (disulfide)
        if any(g.sym(n) == "S" for n in g.neighbors(s)):
            continue
        hn_s = g.heavy_neighbors(s)
        if g.h_count(s) >= 1:
            if any(g.is_aromatic_C(n) for n in hn_s):
                results.append(("thiophenol", (s,) + tuple(hn_s)))
            else:
                results.append(("thiol", (s,) + tuple(hn_s)))
        elif len(hn_s) == 2:
            results.append(("thioether", (s,) + tuple(hn_s)))
    return results


def _match_halide(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    results = []
    for hal in ("F", "Cl", "Br", "I"):
        for x in g.atoms_of(hal):
            c_nbrs = [n for n in g.heavy_neighbors(x) if g.sym(n) == "C"]
            if not c_nbrs:
                continue
            c = c_nbrs[0]
            if g.is_aromatic_C(c):
                results.append(("aryl_halide", (c, x)))
            elif g.has_bond_to(c, "C", 2):
                results.append(("vinyl_halide", (c, x)))
            elif g.has_bond_to(c, "O", 2):
                continue  # acyl halide handled in carbonyl family
            else:
                results.append(("alkyl_halide", (c, x)))
    return results


def _match_si_b(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    results = []
    for si in g.atoms_of("Si"):
        o_nbrs = [n for n in g.neighbors(si) if g.sym(n) == "O"]
        c_nbrs = [n for n in g.neighbors(si) if g.sym(n) == "C"]
        h_nbrs = [n for n in g.neighbors(si) if g.sym(n) == "H"]
        if o_nbrs:
            results.append(("silyl_ether", (si,) + tuple(o_nbrs)))
        if c_nbrs:
            results.append(("organosilane", (si,) + tuple(c_nbrs)))
        if h_nbrs:
            results.append(("silane", (si,) + tuple(h_nbrs)))

    for b in g.atoms_of("B"):
        o_nbrs = [n for n in g.neighbors(b) if g.sym(n) == "O"]
        h_nbrs = [n for n in g.neighbors(b) if g.sym(n) == "H"]
        if o_nbrs:
            if any(g.h_count(o) >= 1 for o in o_nbrs):
                results.append(("boronic_acid", (b,) + tuple(o_nbrs)))
            else:
                results.append(("boronate_ester", (b,) + tuple(o_nbrs)))
        if h_nbrs:
            results.append(("borane", (b,) + tuple(h_nbrs)))

    for p in g.atoms_of("P"):
        h_nbrs = [n for n in g.neighbors(p) if g.sym(n) == "H"]
        if h_nbrs and not g.has_bond_to(p, "O", 2):
            results.append(("phosphine", (p,) + tuple(h_nbrs)))
    return results


def _match_ch_context(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    """Classify C-H bonds by chemical context."""
    results = []
    for c in g.atoms_of("C"):
        h_nbrs = [n for n in g.neighbors(c) if g.sym(n) == "H"]
        if not h_nbrs:
            continue
        h_num = len(h_nbrs)
        atoms = (c,) + tuple(h_nbrs)

        # sp C-H (alkynyl)
        if g.has_bond_to(c, "C", 3) or g.has_bond_to(c, "N", 3):
            results.append(("ch_alkynyl", atoms))
            continue

        # Aromatic C-H
        if g.is_aromatic_C(c):
            if g.has_heteroatom_ring_nbr(c):
                results.append(("ch_heteroaromatic", atoms))
            else:
                results.append(("ch_aromatic", atoms))
            continue

        # Aldehyde C-H (C=O with H)
        if g.has_bond_to(c, "O", 2):
            results.append(("ch_aldehyde", atoms))
            continue

        # Vinyl / sp2 C-H
        if g.has_bond_to(c, "C", 2) or g.has_bond_to(c, "N", 2):
            results.append(("ch_vinyl", atoms))
            continue

        # sp3 C-H with context
        hn_c = g.heavy_neighbors(c)
        adj_elements = {g.sym(n) for n in hn_c}

        suffix = "CH%d" % h_num if h_num <= 3 else "CH"

        if any(g.has_bond_to(n, "O", 2) for n in hn_c if g.sym(n) == "C"):
            results.append(("ch_alpha_carbonyl_%s" % suffix, atoms))
        elif any(g.has_bond_to(n, "S", 2) for n in hn_c if g.sym(n) == "C"):
            results.append(("ch_alpha_thiocarbonyl_%s" % suffix, atoms))
        elif "O" in adj_elements:
            results.append(("ch_alpha_ether_%s" % suffix, atoms))
        elif "N" in adj_elements:
            results.append(("ch_alpha_amine_%s" % suffix, atoms))
        elif "S" in adj_elements:
            results.append(("ch_alpha_thioether_%s" % suffix, atoms))
        elif adj_elements & {"F", "Cl", "Br", "I"}:
            results.append(("ch_alpha_halo_%s" % suffix, atoms))
        elif g.is_in_3ring(c):
            results.append(("ch_cyclopropane", atoms))
        elif h_num == 3:
            results.append(("ch_methyl", atoms))
        elif h_num == 2:
            results.append(("ch_methylene", atoms))
        elif h_num == 1:
            results.append(("ch_methine", atoms))
        else:
            results.append(("ch_sp3", atoms))
    return results


def _match_oh_context(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    """Classify O-H bonds by chemical context."""
    results = []
    for o in g.atoms_of("O"):
        if g.h_count(o) == 0:
            continue
        h_atoms = tuple(n for n in g.neighbors(o) if g.sym(n) == "H")
        hn_o = g.heavy_neighbors(o)

        if any(g.sym(n) == "O" for n in hn_o):
            results.append(("oh_hydroperoxide", (o,) + h_atoms))
            continue
        if any(g.sym(n) == "N" for n in hn_o):
            results.append(("oh_hydroxamic", (o,) + h_atoms))
            continue
        if any(g.sym(n) == "P" for n in hn_o):
            results.append(("oh_phosphoric", (o,) + h_atoms))
            continue
        if any(g.sym(n) == "S" for n in hn_o):
            results.append(("oh_sulfonic", (o,) + h_atoms))
            continue
        if any(g.sym(n) == "B" for n in hn_o):
            results.append(("oh_boronic", (o,) + h_atoms))
            continue

        c_nbr = next((n for n in hn_o if g.sym(n) == "C"), None)
        if c_nbr is not None:
            if g.has_bond_to(c_nbr, "O", 2):
                results.append(("oh_carboxylic", (o, c_nbr) + h_atoms))
            elif g.is_aromatic_C(c_nbr):
                results.append(("oh_phenol", (o, c_nbr) + h_atoms))
            elif g.has_bond_to(c_nbr, "C", 2):
                results.append(("oh_enol", (o, c_nbr) + h_atoms))
            else:
                results.append(("oh_alcohol", (o, c_nbr) + h_atoms))
            continue

        results.append(("oh_alcohol", (o,) + h_atoms))
    return results


def _match_nh_context(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    """Classify N-H bonds by chemical context."""
    results = []
    for n in g.atoms_of("N"):
        if g.h_count(n) == 0:
            continue
        h_atoms = tuple(nb for nb in g.neighbors(n) if g.sym(nb) == "H")
        hn_n = g.heavy_neighbors(n)

        if g.is_pyrrole_N(n):
            results.append(("nh_pyrrole", (n,) + h_atoms))
            continue

        # Amide N-H: N bonded to C(=O)
        c_co = [nb for nb in hn_n if g.sym(nb) == "C" and g.has_bond_to(nb, "O", 2)]
        if c_co:
            c = c_co[0]
            n_count = sum(1 for nb2 in g.heavy_neighbors(c) if g.sym(nb2) == "N")
            has_co_o = any(g.sym(nb2) == "O" and g.bo(c, nb2) == 1 for nb2 in g.heavy_neighbors(c))
            if n_count >= 2:
                results.append(("nh_urea", (n, c) + h_atoms))
            elif has_co_o:
                results.append(("nh_carbamate", (n, c) + h_atoms))
            else:
                results.append(("nh_amide", (n, c) + h_atoms))
            continue

        # Sulfonamide N-H
        if any(g.sym(nb) == "S" and g.has_bond_to(nb, "O", 2) for nb in hn_n):
            results.append(("nh_sulfonamide", (n,) + h_atoms))
            continue

        # Hydrazine/hydrazone N-H
        if any(g.sym(nb) == "N" for nb in hn_n):
            results.append(("nh_hydrazine", (n,) + h_atoms))
            continue

        # Guanidine/amidine N-H
        cn_n = [nb for nb in hn_n if g.sym(nb) == "C"]
        if cn_n and sum(1 for nb2 in g.heavy_neighbors(cn_n[0]) if g.sym(nb2) == "N") >= 2:
            results.append(("nh_guanidine", (n,) + h_atoms))
            continue

        hc_n = g.h_count(n)
        if hc_n >= 2:
            if any(g.is_aromatic_C(nb) for nb in hn_n):
                results.append(("nh_aniline_primary", (n,) + h_atoms))
            else:
                results.append(("nh_primary_amine", (n,) + h_atoms))
        else:
            results.append(("nh_secondary_amine", (n,) + h_atoms))
    return results


def _match_sh_ph_sih_bh(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    """S-H, P-H, Si-H, B-H bonds."""
    results = []
    for s in g.atoms_of("S"):
        h_atoms = tuple(n for n in g.neighbors(s) if g.sym(n) == "H")
        if h_atoms:
            if any(g.is_aromatic_C(n) for n in g.heavy_neighbors(s)):
                results.append(("sh_thiophenol", (s,) + h_atoms))
            else:
                results.append(("sh_thiol", (s,) + h_atoms))
    # P-H, Si-H, B-H already handled in _match_si_b
    return results


def _match_cc_single(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    """C-C single bonds: cyclopropane vs alkane."""
    results = []
    seen = set()
    for c1 in g.atoms_of("C"):
        for c2 in g.neighbors_with(c1, "C", 1):
            pair = (min(c1, c2), max(c1, c2))
            if pair in seen:
                continue
            seen.add(pair)
            if g.is_in_3ring(c1) and g.is_in_3ring(c2):
                results.append(("cyclopropane", pair))
            else:
                results.append(("alkane_cc", pair))
    return results


def _match_co_single(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    """C-O single bonds not covered by alcohol/ether/ester matchers."""
    results = []
    for o in g.atoms_of("O"):
        c_nbrs = [n for n in g.neighbors(o) if g.sym(n) == "C" and g.bo(o, n) == 1]
        if not c_nbrs:
            continue
        for c in c_nbrs:
            if g.has_bond_to(c, "O", 2):
                results.append(("co_ester", (c, o)))
    return results


def _match_cn_single(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    """C-N single bonds with context."""
    results = []
    for n in g.atoms_of("N"):
        c_nbrs = [nb for nb in g.neighbors(n) if g.sym(nb) == "C" and g.bo(n, nb) == 1]
        if not c_nbrs:
            continue
        hn_n = g.heavy_neighbors(n)
        hc_n = g.h_count(n)

        for c in c_nbrs:
            if g.is_aromatic_C(c) and hc_n >= 1:
                results.append(("cn_aniline", (c, n)))
            elif any(g.has_bond_to(nb2, "O", 2) for nb2 in hn_n if g.sym(nb2) == "C"):
                c2 = next(nb2 for nb2 in hn_n if g.sym(nb2) == "C" and g.has_bond_to(nb2, "O", 2))
                if any(g.sym(nb3) == "O" and g.bo(c2, nb3) == 1 for nb3 in g.heavy_neighbors(c2)):
                    results.append(("cn_carbamate", (c, n)))
                else:
                    results.append(("cn_amide", (c, n)))
            elif any(g.sym(nb) == "S" and g.has_bond_to(nb, "O", 2) for nb in hn_n):
                results.append(("cn_sulfonamide", (c, n)))
            elif hc_n >= 2:
                results.append(("cn_primary_amine", (c, n)))
            elif hc_n == 1:
                results.append(("cn_secondary_amine", (c, n)))
            else:
                results.append(("cn_tertiary_amine", (c, n)))
    return results


def _match_cs_single(g: MolGraph) -> List[Tuple[str, Tuple[int, ...]]]:
    """C-S single bonds with context."""
    results = []
    for s in g.atoms_of("S"):
        c_nbrs = [n for n in g.neighbors(s) if g.sym(n) == "C" and g.bo(s, n) == 1]
        if not c_nbrs:
            continue
        for c in c_nbrs:
            if g.h_count(s) >= 1:
                results.append(("cs_thiol", (c, s)))
            elif any(g.sym(n) == "S" for n in g.heavy_neighbors(s)):
                results.append(("cs_disulfide", (c, s)))
            elif g.has_bond_to(s, "O", 2):
                results.append(("cs_sulfoxide", (c, s)))
            elif g.has_bond_to(s, "C", 3):
                results.append(("cs_thiocyanate", (c, s)))
            else:
                results.append(("cs_thioether", (c, s)))
    return results


# ============================================================================
# All pattern matchers in priority order (more specific first)
# ============================================================================

_ALL_MATCHERS = [
    # Complex multi-atom groups first (most specific)
    _match_carbonyl_family,
    _match_thiocarbonyl,
    _match_no_bonds,
    _match_so_bonds,
    _match_po_bonds,
    _match_nn_bonds,
    _match_oo_bonds,
    _match_ss_bonds,
    _match_triple_bonds,
    _match_cc_double,
    _match_cn_double,
    _match_si_b,
    _match_halide,
    # X-H bonds (mode-level context)
    _match_oh_context,
    _match_nh_context,
    _match_ch_context,
    _match_sh_ph_sih_bh,
    # Single bonds
    _match_alcohol_ether,
    _match_amine,
    _match_thiol_thioether,
    _match_co_single,
    _match_cn_single,
    _match_cs_single,
    _match_cc_single,
]


# ============================================================================
# Core engine: find all matches, resolve hierarchy
# ============================================================================


def _find_all_matches(symbols: list, topology: dict) -> Dict[str, List[Tuple[int, ...]]]:
    """Run all FG pattern matchers against the molecular graph."""
    g = MolGraph(symbols, topology)
    raw: Dict[str, List[Tuple[int, ...]]] = {}

    for matcher in _ALL_MATCHERS:
        for name, atoms in matcher(g):
            raw.setdefault(name, []).append(atoms)

    return raw


def _resolve_hierarchy(
    matches: Dict[str, List[Tuple[int, ...]]],
) -> Dict[str, List[Tuple[int, ...]]]:
    """AccFG-style hierarchy resolution.

    Remove functional groups whose atom sets are strict subsets of a
    more-specific group.  This ensures e.g. a standalone "ketone" (C,O)
    does not also appear alongside "ester" (C,O,O') which already
    contains that C=O.
    """
    if len(matches) <= 1:
        return dict(matches)

    # Build flat list: (name, frozenset_of_atoms)
    entries = []
    for name, instances in matches.items():
        for atoms in instances:
            entries.append((name, frozenset(atoms)))

    # For each entry, check if its atoms are a strict subset of another entry
    subsumed_instances: Set[Tuple[str, frozenset]] = set()
    for i, (name_a, set_a) in enumerate(entries):
        for j, (name_b, set_b) in enumerate(entries):
            if i == j or name_a == name_b:
                continue
            if set_a < set_b:  # strict subset
                subsumed_instances.add((name_a, set_a))

    # Rebuild, excluding subsumed instances
    result: Dict[str, List[Tuple[int, ...]]] = {}
    for name, instances in matches.items():
        kept = []
        for atoms in instances:
            if (name, frozenset(atoms)) not in subsumed_instances:
                kept.append(atoms)
        if kept:
            result[name] = kept

    return result


# ============================================================================
# Public API
# ============================================================================


def identify_functional_groups(
    symbols: list,
    topology: dict,
) -> Dict[str, List[Tuple[int, ...]]]:
    """Identify all functional groups in a molecule.

    Uses AccFG-style graph pattern matching on the molecular topology,
    with hierarchical deduplication to keep only the most specific groups.

    Args:
        symbols: List of element symbols for each atom.
        topology: Topology dict from ``_build_topology`` (needs
            ``neighbors`` and ``bond_orders``).

    Returns:
        Dictionary mapping FG name → list of atom index tuples.
        Each tuple contains the indices of atoms in that FG instance.
    """
    raw = _find_all_matches(symbols, topology)
    return _resolve_hierarchy(raw)


# ============================================================================
# Label formatting: internal name → display label matching ir.py convention
# ============================================================================

_LABEL_FORMAT = {
    # Carbonyl family
    "ketone": "C=O (ketone)",
    "aldehyde": "C=O (aldehyde)",
    "carboxylic_acid": "C=O (carboxylic acid)",
    "ester": "C=O (ester)",
    "lactone": "C=O (lactone)",
    "amide": "C=O (amide)",
    "lactam": "C=O (lactam)",
    "anhydride": "C=O (anhydride)",
    "carbonate": "C=O (carbonate)",
    "carbamate": "C=O (carbamate/urethane)",
    "acyl_halide": "C=O (acyl halide)",
    "urea": "C=O (urea)",
    "thioester": "C=O (thioester)",
    # Thiocarbonyl
    "thioamide": "C=S (thioamide)",
    "thioketone": "C=S (thiocarbonyl/thioketone)",
    # C=C
    "aromatic_ring": "C=C (aromatic)",
    "heteroaromatic_ring": "C=C (heteroaromatic)",
    "alkene": "C=C (alkene)",
    "allene": "C=C (allene/cumulene)",
    "enamine": "C=C (enamine)",
    "enol_ether": "C=C (enol/vinyl ether)",
    # C=N
    "imine": "C=N (imine/Schiff base)",
    "imine_ring": "C=N (imine, ring)",
    "oxime": "C=N (oxime)",
    "hydrazone": "C=N (hydrazone)",
    # Triple bonds
    "nitrile": "C=N (nitrile)",
    "alkyne": "C=C (alkyne)",
    # N-N
    "diazonium": "N=N (diazonium)",
    "azo": "N=N (azo)",
    "hydrazine": "N-N (hydrazine)",
    # N-O
    "nitro": "N=O (nitro group)",
    "nitroso": "N=O (nitroso)",
    "n_oxide": "N-O (N-oxide)",
    "hydroxylamine": "N-O (hydroxylamine)",
    # O-O
    "hydroperoxide": "O-O (hydroperoxide)",
    "peroxide": "O-O (peroxide)",
    # S-S
    "disulfide": "S-S (disulfide)",
    # S=O
    "sulfoxide": "S=O (sulfoxide)",
    "sulfone": "S=O (sulfone)",
    "sulfonic_acid": "S=O (sulfonic acid)",
    "sulfonamide": "S=O (sulfonamide)",
    # P=O / P-O
    "phosphonamide": "P=O (phosphonamide)",
    "phosphonic_acid": "P=O (phosphonic acid)",
    "phosphate": "P=O (phosphate/phosphonate)",
    "phosphate_ester": "P-O (phosphate ester)",
    # Alcohol / ether / epoxide
    "alcohol": "C-O (alcohol)",
    "phenol": "C-O (phenol)",
    "enol": "C-O (enol)",
    "ether": "C-O (ether)",
    "epoxide": "C-O (epoxide)",
    "vinyl_ether": "C-O (vinyl ether/enol ether)",
    # Amine
    "primary_amine": "C-N (primary amine)",
    "secondary_amine": "C-N (secondary amine)",
    "tertiary_amine": "C-N (tertiary amine)",
    "aniline": "C-N (aniline/aromatic amine)",
    "guanidine": "N-H (guanidine/amidine)",
    "pyrrole_n": "N-H (pyrrole/indole)",
    # Thiol / thioether
    "thiol": "C-S (thiol)",
    "thiophenol": "C-S (thiophenol)",
    "thioether": "C-S (thioether)",
    # Halides
    "aryl_halide": "C-X (aryl halide)",
    "vinyl_halide": "C-X (vinyl halide)",
    "alkyl_halide": "C-X (alkyl halide)",
    # Si, B, P
    "silyl_ether": "Si-O (silyl ether/siloxane)",
    "organosilane": "Si-C (organosilane)",
    "silane": "Si-H (silane)",
    "boronic_acid": "B-O (boronic acid)",
    "boronate_ester": "B-O (boronate ester)",
    "borane": "B-H (borane)",
    "phosphine": "P-H (phosphine)",
    # C-C single
    "cyclopropane": "C-C (cyclopropane)",
    "alkane_cc": "C-C (alkane)",
    # O-H context
    "oh_hydroperoxide": "O-H (hydroperoxide)",
    "oh_hydroxamic": "O-H (hydroxamic acid / oxime)",
    "oh_phosphoric": "O-H (phosphoric acid)",
    "oh_sulfonic": "O-H (sulfonic acid)",
    "oh_boronic": "O-H (boronic acid)",
    "oh_carboxylic": "O-H (carboxylic acid)",
    "oh_phenol": "O-H (phenol)",
    "oh_enol": "O-H (enol)",
    "oh_alcohol": "O-H (alcohol)",
    # N-H context
    "nh_pyrrole": "N-H (pyrrole/indole)",
    "nh_urea": "N-H (urea)",
    "nh_carbamate": "N-H (carbamate/urethane)",
    "nh_amide": "N-H (amide)",
    "nh_sulfonamide": "N-H (sulfonamide)",
    "nh_hydrazine": "N-H (hydrazine/hydrazone)",
    "nh_guanidine": "N-H (guanidine/amidine)",
    "nh_aniline_primary": "N-H (aniline, primary)",
    "nh_primary_amine": "N-H (primary amine)",
    "nh_secondary_amine": "N-H (secondary amine)",
    # C-H context
    "ch_alkynyl": "C-H (sp, alkynyl)",
    "ch_heteroaromatic": "C-H (heteroaromatic)",
    "ch_aromatic": "C-H (aromatic)",
    "ch_aldehyde": "C-H (aldehyde)",
    "ch_vinyl": "C-H (vinyl/sp2)",
    "ch_methyl": "C-H (CH3, methyl)",
    "ch_methylene": "C-H (CH2, methylene)",
    "ch_methine": "C-H (CH, methine)",
    "ch_sp3": "C-H (sp3)",
    "ch_cyclopropane": "C-H (CH2, cyclopropane)",
    "ch_alpha_carbonyl_CH3": "C-H (CH3, alpha-carbonyl)",
    "ch_alpha_carbonyl_CH2": "C-H (CH2, alpha-carbonyl)",
    "ch_alpha_carbonyl_CH1": "C-H (CH1, alpha-carbonyl)",
    "ch_alpha_thiocarbonyl_CH3": "C-H (CH3, alpha-thiocarbonyl)",
    "ch_alpha_ether_CH3": "C-H (CH3, alpha-ether/alcohol)",
    "ch_alpha_ether_CH2": "C-H (CH2, alpha-ether/alcohol)",
    "ch_alpha_ether_CH1": "C-H (CH1, alpha-ether/alcohol)",
    "ch_alpha_amine_CH3": "C-H (CH3, alpha-amine)",
    "ch_alpha_amine_CH2": "C-H (CH2, alpha-amine)",
    "ch_alpha_amine_CH1": "C-H (CH1, alpha-amine)",
    "ch_alpha_thioether_CH3": "C-H (CH3, alpha-thioether)",
    "ch_alpha_thioether_CH2": "C-H (CH2, alpha-thioether)",
    "ch_alpha_halo_CH3": "C-H (CH3, alpha-halo)",
    "ch_alpha_halo_CH2": "C-H (CH2, alpha-halo)",
    # S-H
    "sh_thiophenol": "S-H (thiophenol)",
    "sh_thiol": "S-H (thiol)",
    # C-O single (from co_single matcher)
    "co_ester": "C-O (ester)",
    # C-N single
    "cn_aniline": "C-N (aniline/aromatic amine)",
    "cn_carbamate": "C-N (carbamate)",
    "cn_amide": "C-N (amide)",
    "cn_sulfonamide": "C-N (sulfonamide)",
    "cn_primary_amine": "C-N (primary amine)",
    "cn_secondary_amine": "C-N (secondary amine)",
    "cn_tertiary_amine": "C-N (tertiary amine)",
    # C-S single
    "cs_thiol": "C-S (thiol)",
    "cs_disulfide": "C-S (disulfide)",
    "cs_sulfoxide": "C-S (sulfoxide/sulfone)",
    "cs_thiocyanate": "C-S (thiocyanate)",
    "cs_thioether": "C-S (thioether)",
}


def _bond_prefix(atom_indices: list, symbols: list, topology: dict) -> Optional[str]:
    """Derive bond label (e.g. 'C=O', 'O-H') from a mode's dominant atoms."""
    if len(atom_indices) < 2 or not symbols or not topology:
        return None
    i, j = int(atom_indices[0]), int(atom_indices[1])
    if i >= len(symbols) or j >= len(symbols):
        return None
    bo_map = topology.get("bond_orders", {})
    order = bo_map.get((min(i, j), max(i, j)), 1)
    si, sj = symbols[i], symbols[j]
    sym = _BO_SYM.get(order, "-")
    return "%s%s%s" % (si, sym, sj)


def get_functional_group_for_atoms(
    fg_map: Dict[str, List[Tuple[int, ...]]],
    atom_indices: list,
    symbols: Optional[list] = None,
    topology: Optional[dict] = None,
) -> Optional[str]:
    """Map a vibrational mode's dominant atoms to the best functional group.

    Finds the FG whose atoms have the highest overlap with the mode's
    dominant atoms, then returns a formatted label matching the ir.py
    convention (e.g. ``"C=O (ketone)"``).

    The bond prefix (e.g. ``C=O``, ``O-H``, ``C-N``) is derived from the
    mode's dominant atoms so that the label reflects the actual vibrating
    bond, not just the FG's canonical bond.

    Args:
        fg_map: Output of :func:`identify_functional_groups`.
        atom_indices: Dominant atom indices for a vibrational mode.
        symbols: Element symbols (used for bond prefix and halide formatting).
        topology: Topology dict (used for bond order lookup).

    Returns:
        Formatted FG label string, or ``None`` if no match found.
    """
    if not fg_map or not atom_indices:
        return None

    dominant = set(int(a) for a in atom_indices)
    best_fg = None
    best_score = 0
    best_atoms = None

    for fg_name, instances in fg_map.items():
        for atom_tuple in instances:
            fg_set = set(atom_tuple)
            overlap = len(dominant & fg_set)
            if overlap > best_score:
                best_score = overlap
                best_fg = fg_name
                best_atoms = atom_tuple
            elif overlap == best_score and overlap > 0:
                # Prefer more specific (larger) FG on tie
                if best_atoms is not None and len(atom_tuple) > len(best_atoms):
                    best_fg = fg_name
                    best_atoms = atom_tuple

    if best_fg is None:
        return None

    # Extract the FG context name (the part in parentheses)
    label = _LABEL_FORMAT.get(best_fg)
    if label is None:
        return best_fg.replace("_", " ")

    # Fix halide labels with actual element
    if "C-X" in label and symbols is not None and best_atoms is not None:
        for a in best_atoms:
            if symbols[a] in ("F", "Cl", "Br", "I"):
                label = label.replace("C-X", "C-%s" % symbols[a])
                break

    # Derive bond prefix from dominant atoms and replace the default one
    bond_pfx = _bond_prefix(atom_indices, symbols, topology)
    if bond_pfx and "(" in label:
        # label is like "C=O (ketone)" → replace prefix with actual bond
        paren_start = label.index("(")
        context = label[paren_start:]  # "(ketone)"
        return "%s %s" % (bond_pfx, context)

    return label
