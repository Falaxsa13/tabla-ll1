# app.py

import streamlit as st
import pandas as pd
import graphviz
import re


# ——————————————————————————
# 1. Scanner y parseo de la gramática
# ——————————————————————————
def scan_grammar(text, empty_sym):
    """
    Tokeniza cada línea de la gramática en símbolos:
    tokens: '->', '|', símbolo vacío, identificadores alfanuméricos.
    """
    rules = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Regex para flecha, barra, símbolo vacío o identificadores
        pattern = r"->|\||%s|[A-Za-z0-9_]+" % re.escape(empty_sym)
        tokens = re.findall(pattern, line)
        rules.append(tokens)
    return rules


def parse_grammar_with_scanner(text, empty_sym):
    raw_rules = scan_grammar(text, empty_sym)
    grammar = {}
    for tokens in raw_rules:
        # tokens: [lhs, '->', sym1, sym2, '|', sym3, ...]
        lhs = tokens[0]
        # Validación de convención: LHS en minúscula
        if not lhs.islower():
            raise ValueError(f"El LHS `{lhs}` debe ser minúscula (no-terminal)")
        prods = []
        curr = []
        for tok in tokens[2:]:
            if tok == "|":
                prods.append(curr)
                curr = []
            else:
                curr.append(tok)
        prods.append(curr)
        grammar[lhs] = prods
    return grammar


# ——————————————————————————
# 2. Algoritmo FIRST
# ——————————————————————————
def compute_first(grammar):
    first = {nt: set() for nt in grammar}
    changed = True
    while changed:
        changed = False
        for A, prods in grammar.items():
            for prod in prods:
                all_nullable = True
                for sym in prod:
                    if sym == empty_sym_input:
                        if empty_sym_input not in first[A]:
                            first[A].add(empty_sym_input)
                            changed = True
                        all_nullable = False
                        break
                    if sym not in grammar:  # terminal
                        if sym not in first[A]:
                            first[A].add(sym)
                            changed = True
                        all_nullable = False
                        break
                    # no terminal
                    for s in first[sym] - {empty_sym_input}:
                        if s not in first[A]:
                            first[A].add(s)
                            changed = True
                    if empty_sym_input not in first[sym]:
                        all_nullable = False
                        break
                if all_nullable and empty_sym_input not in first[A]:
                    first[A].add(empty_sym_input)
                    changed = True
    return first


# ——————————————————————————
# 3. Algoritmo FOLLOW
# ——————————————————————————
def compute_follow(grammar, first):
    follow = {nt: set() for nt in grammar}
    start = next(iter(grammar))
    follow[start].add("$")
    changed = True
    while changed:
        changed = False
        for A, prods in grammar.items():
            for prod in prods:
                for i, B in enumerate(prod):
                    if B in grammar:
                        beta = prod[i + 1 :]
                        first_beta = set()
                        nullable = True
                        for sym in beta:
                            if sym == empty_sym_input:
                                first_beta.add(empty_sym_input)
                                continue
                            if sym not in grammar:
                                first_beta.add(sym)
                                nullable = False
                                break
                            first_beta |= first[sym] - {empty_sym_input}
                            if empty_sym_input in first[sym]:
                                nullable = True
                            else:
                                nullable = False
                                break
                        # añade FIRST(beta) \ {ε} a FOLLOW(B)
                        for t in first_beta - {empty_sym_input}:
                            if t not in follow[B]:
                                follow[B].add(t)
                                changed = True
                        # si beta vacío o nullable, añade FOLLOW(A)
                        if not beta or nullable:
                            for t in follow[A]:
                                if t not in follow[B]:
                                    follow[B].add(t)
                                    changed = True
    return follow


# ——————————————————————————
# 4. Construir tabla LL(1)
# ——————————————————————————
def compute_first_of_string(symbols, grammar, first):
    result = set()
    nullable = True
    for sym in symbols:
        if sym == empty_sym_input:
            result.add(empty_sym_input)
            break
        if sym not in grammar:
            result.add(sym)
            nullable = False
            break
        result |= first[sym] - {empty_sym_input}
        if empty_sym_input in first[sym]:
            continue
        else:
            nullable = False
            break
    if nullable:
        result.add(empty_sym_input)
    return result


def compute_parse_table(grammar, first, follow):
    terminals = sorted(
        {t for prods in grammar.values() for prod in prods for t in prod if t not in grammar and t != empty_sym_input}
    )
    terminals.append("$")
    table = {A: {t: None for t in terminals} for A in grammar}
    for A, prods in grammar.items():
        for prod in prods:
            first_alpha = compute_first_of_string(prod, grammar, first)
            for t in first_alpha - {empty_sym_input}:
                table[A][t] = prod
            if empty_sym_input in first_alpha:
                for t in follow[A]:
                    table[A][t] = prod
    return table, terminals


# ——————————————————————————
# 5. Simulación de parsing LL(1)
# ——————————————————————————
def simulate_ll1(grammar, table, tokens, start):
    stack = ["$", start]
    tokens = tokens + ["$"]
    i = 0
    trace = []
    while True:
        top = stack.pop()
        current = tokens[i]
        trace.append({"Stack": " ".join(stack + [top]), "Input": " ".join(tokens[i:]), "Action": ""})
        if top in grammar:
            prod = table[top].get(current)
            if not prod:
                trace[-1]["Action"] = f"Error: no regla para {top} con {current}"
                break
            trace[-1]["Action"] = f"{top} → {' '.join(prod)}"
            if prod != [empty_sym_input]:
                for sym in reversed(prod):
                    stack.append(sym)
        else:
            if top == current:
                trace[-1]["Action"] = f"Match {current}"
                i += 1
            else:
                trace[-1]["Action"] = f"Error: expected {top}, got {current}"
                break
        if top == "$" and current == "$":
            trace[-1]["Action"] = "Aceptado"
            break
        if len(trace) > 1000:
            break
    return trace


# ——————————————————————————
# 6. Construir Árbol de Derivación
# ——————————————————————————
class Node:
    def __init__(self, symbol):
        self.symbol = symbol
        self.children = []


def build_parse_tree(grammar, table, tokens, start):
    tokens = tokens + ["$"]
    i = 0

    def recurse(sym):
        nonlocal i
        node = Node(sym)
        if sym not in grammar:
            if sym == tokens[i]:
                i += 1
            return node
        prod = table[sym].get(tokens[i])
        if not prod:
            return node
        for s in prod:
            if s == empty_sym_input:
                node.children.append(Node(empty_sym_input))
            else:
                child = recurse(s)
                node.children.append(child)
        return node

    return recurse(start)


def tree_to_dot(node, dot=None):
    if dot is None:
        dot = graphviz.Digraph()
    uid = str(id(node))
    dot.node(uid, node.symbol)
    for child in node.children:
        cid = str(id(child))
        dot.node(cid, child.symbol)
        dot.edge(uid, cid)
        tree_to_dot(child, dot)
    return dot


# ——————————————————————————
# 7. Interfaz Streamlit
# ——————————————————————————
st.title("📚 Simulador LL(1) Completo con Scanner")
st.write("Ingresa tu gramática, define el símbolo vacío, y observa FIRST, FOLLOW, tabla, traza y árbol.")

empty_sym_input = st.text_input("Símbolo para cadena vacía", "ε")
grammar_input = st.text_area("📝 Gramática (usa `->` y `|`)", height=200)
if not grammar_input.strip():
    st.stop()

if st.button("▶️ Procesar Gramática"):
    try:
        grammar = parse_grammar_with_scanner(grammar_input, empty_sym_input)
        first = compute_first(grammar)
        follow = compute_follow(grammar, first)
        table, terminals = compute_parse_table(grammar, first, follow)
        start = next(iter(grammar))

        # Mostrar FIRST & FOLLOW
        st.subheader("🎯 Conjuntos FIRST & FOLLOW")
        for A in grammar:
            st.write(
                f"**FIRST({A})** = {{ {', '.join(sorted(first[A]))} }} | "
                f"**FOLLOW({A})** = {{ {', '.join(sorted(follow[A]))} }}"
            )

        # Tabla LL(1)
        st.subheader("📋 Tabla LL(1)")
        df = pd.DataFrame(index=grammar.keys(), columns=terminals)
        for A in grammar:
            for t in terminals:
                prod = table[A].get(t)
                df.loc[A, t] = " ".join(prod) if prod else ""
        df.index.name = "Nonterminal"
        df.insert(0, "FOLLOW", [", ".join(sorted(follow[A])) for A in df.index])
        df.insert(0, "FIRST", [", ".join(sorted(first[A])) for A in df.index])
        st.dataframe(df)

        # Simulación y Árbol
        col1, col2 = st.columns(2)
        token_input = st.text_input("Tokens separados por espacio", "id + id")
        if st.button("▶️ Simular y Mostrar Árbol"):
            tokens = token_input.split()
            trace = simulate_ll1(grammar, table, tokens, start)
            with col1:
                st.subheader("🚀 Simulación LL(1)")
                st.dataframe(pd.DataFrame(trace))
            tree = build_parse_tree(grammar, table, tokens, start)
            dot = tree_to_dot(tree)
            with col2:
                st.subheader("🌳 Árbol de Derivación")
                st.graphviz_chart(dot)
    except Exception as e:
        st.error(f"Error al procesar la gramática: {e}")
