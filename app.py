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
def compute_first(grammar, empty_sym):
    first = {nt: set() for nt in grammar}
    changed = True
    while changed:
        changed = False
        for A, prods in grammar.items():
            for prod in prods:
                all_nullable = True
                for sym in prod:
                    if sym == empty_sym:
                        if empty_sym not in first[A]:
                            first[A].add(empty_sym)
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
                    for s in first[sym] - {empty_sym}:
                        if s not in first[A]:
                            first[A].add(s)
                            changed = True
                    if empty_sym not in first[sym]:
                        all_nullable = False
                        break
                if all_nullable and empty_sym not in first[A]:
                    first[A].add(empty_sym)
                    changed = True
    return first


# ——————————————————————————
# 3. Algoritmo FOLLOW
# ——————————————————————————
def compute_follow(grammar, first, empty_sym):
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
                            if sym == empty_sym:
                                first_beta.add(empty_sym)
                                continue
                            if sym not in grammar:
                                first_beta.add(sym)
                                nullable = False
                                break
                            first_beta |= first[sym] - {empty_sym}
                            if empty_sym in first[sym]:
                                nullable = True
                            else:
                                nullable = False
                                break
                        # añade FIRST(beta) \ {ε} a FOLLOW(B)
                        for t in first_beta - {empty_sym}:
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
def compute_first_of_string(symbols, grammar, first, empty_sym):
    result = set()
    nullable = True
    for sym in symbols:
        if sym == empty_sym:
            result.add(empty_sym)
            break
        if sym not in grammar:
            result.add(sym)
            nullable = False
            break
        result |= first[sym] - {empty_sym}
        if empty_sym in first[sym]:
            continue
        else:
            nullable = False
            break
    if nullable:
        result.add(empty_sym)
    return result


def compute_parse_table(grammar, first, follow, empty_sym):
    terminals = sorted(
        {t for prods in grammar.values() for prod in prods for t in prod if t not in grammar and t != empty_sym}
    )
    terminals.append("$")
    table = {A: {t: None for t in terminals} for A in grammar}
    for A, prods in grammar.items():
        for prod in prods:
            first_alpha = compute_first_of_string(prod, grammar, first, empty_sym)
            for t in first_alpha - {empty_sym}:
                table[A][t] = prod
            if empty_sym in first_alpha:
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


def build_parse_tree(grammar, table, tokens, start, empty_sym):
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
            if s == empty_sym:
                node.children.append(Node(empty_sym))
            else:
                child = recurse(s)
                node.children.append(child)
        return node

    return recurse(start)


def tree_to_dot(node, dot=None):
    if dot is None:
        dot = graphviz.Digraph(node_attr={"style": "filled", "shape": "box"})
    uid = str(id(node))

    # Determinar estilo del nodo según su tipo
    if node.symbol.islower() and node.symbol != empty_sym_input:  # No-terminal
        dot.node(uid, node.symbol, fillcolor="lightblue")
    elif node.symbol == empty_sym_input:  # Símbolo vacío
        dot.node(uid, node.symbol, fillcolor="lightyellow")
    elif node.symbol == "$":  # Fin de cadena
        dot.node(uid, node.symbol, fillcolor="lightgrey")
    else:  # Terminal
        dot.node(uid, node.symbol, fillcolor="lightgreen")

    for child in node.children:
        cid = str(id(child))
        dot.edge(uid, cid)
        tree_to_dot(child, dot)
    return dot


# ——————————————————————————
# 7. Estilos e inicialización
# ——————————————————————————
def set_page_config():
    st.set_page_config(page_title="Simulador LL(1)", page_icon="📚", layout="wide", initial_sidebar_state="expanded")


def apply_custom_css():
    st.markdown(
        """
    <style>
    .main {x
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8cff;
        color: white;
    }
    .small-font {
        font-size: 0.9rem;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #4e8cff;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff8e6;
        border-left: 4px solid #ffcc00;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #e6fff2;
        border-left: 4px solid #00cc66;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
    }
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        overflow: hidden;
    }
    .highlight {
        color: #4e8cff;
        font-weight: bold;
    }
    .card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin-bottom: 1rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def create_info_box(text, box_type="info"):
    st.markdown(f'<div class="{box_type}-box">{text}</div>', unsafe_allow_html=True)


def create_card(title, content):
    st.markdown(
        f"""
    <div class="card">
        <h3>{title}</h3>
        {content}
    </div>
    """,
        unsafe_allow_html=True,
    )


# Ejemplos de gramáticas para el usuario
EXAMPLE_GRAMMARS = {
    "Expresiones Aritméticas": """expr -> term expr_tail
expr_tail -> + term expr_tail | - term expr_tail | ε
term -> factor term_tail
term_tail -> * factor term_tail | / factor term_tail | ε
factor -> ( expr ) | id | num""",
    "Declaraciones Simples": """statement -> if_stmt | assign_stmt | while_stmt
if_stmt -> if ( expr ) statement else statement
assign_stmt -> id = expr ;
while_stmt -> while ( expr ) statement
expr -> id""",
    "Gramática Simple": """s -> a A
A -> b A | ε
a -> id
b -> num""",
}


# ——————————————————————————
# 8. Interfaz Streamlit Principal
# ——————————————————————————
def main():
    set_page_config()
    apply_custom_css()

    st.title("📚 Simulador LL(1) - Hans Ibarra")
    st.markdown(
        """
    <p class="small-font">
        Herramienta educativa para analizar gramáticas LL(1), calcular conjuntos FIRST y FOLLOW, 
        construir tablas de análisis sintáctico y simular el proceso de parsing.
        Integrantes: Jose Galvez, Jorge Melgarejo, Hans Ibarra.
    </p>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        empty_sym_input = st.text_input("Símbolo para cadena vacía:", "ε", help="Símbolo que representa ε (epsilon)")

        st.header("📋 Ejemplos de Gramáticas")
        example_choice = st.radio(
            "Selecciona un ejemplo:", list(EXAMPLE_GRAMMARS.keys()) + ["Personalizada"], index=len(EXAMPLE_GRAMMARS)
        )

        if st.sidebar.button("ℹ️ Acerca de"):
            st.sidebar.info(
                """
            **Simulador LL(1)**
            
            Esta herramienta está diseñada para ayudar a comprender el
            análisis sintáctico LL(1) con todas sus etapas.
            
            Desarrollado como proyecto educativo para teoría de compiladores.
            """
            )

    # Main area
    tab1, tab2 = st.tabs(["💻 Entrada y Análisis", "📊 Resultados"])

    with tab1:
        st.subheader("📝 Gramática")
        if example_choice == "Personalizada":
            create_info_box(
                """
            <b>Formato de la gramática:</b><br>
            - Usa <code>-></code> para producciones
            - Usa <code>|</code> para alternativas
            - Los no-terminales deben estar en minúsculas
            - Los terminales pueden ser cualquier símbolo (excepto el símbolo vacío)
            Ejemplo: <code>expr -> term + factor | term</code>
            """,
                "info",
            )
            grammar_input = st.text_area(
                "Introduce tu gramática:", height=200, placeholder="Ejemplo:\nexpr -> term + term\nterm -> id | num"
            )
        else:
            grammar_input = st.text_area("Ejemplo de gramática:", value=EXAMPLE_GRAMMARS[example_choice], height=200)

        col1, col2 = st.columns([3, 1])
        with col1:
            process_grammar = st.button("▶️ Procesar Gramática", type="primary", use_container_width=True)
        with col2:
            clear_results = st.button("🗑️ Limpiar", use_container_width=True)

        if clear_results:
            st.experimental_rerun()

    with tab2:
        if not grammar_input.strip():
            create_info_box(
                "Introduce una gramática en la pestaña de Entrada y Análisis para ver los resultados.", "warning"
            )
            st.stop()

        if process_grammar or "grammar" in st.session_state:
            try:
                # Guardar resultados en session_state para mantenerlos entre pestañas
                if process_grammar or "grammar" not in st.session_state:
                    grammar = parse_grammar_with_scanner(grammar_input, empty_sym_input)
                    first = compute_first(grammar, empty_sym_input)
                    follow = compute_follow(grammar, first, empty_sym_input)
                    table, terminals = compute_parse_table(grammar, first, follow, empty_sym_input)
                    start = next(iter(grammar))

                    # Guardar resultados
                    st.session_state.grammar = grammar
                    st.session_state.first = first
                    st.session_state.follow = follow
                    st.session_state.table = table
                    st.session_state.terminals = terminals
                    st.session_state.start = start
                else:
                    # Recuperar resultados
                    grammar = st.session_state.grammar
                    first = st.session_state.first
                    follow = st.session_state.follow
                    table = st.session_state.table
                    terminals = st.session_state.terminals
                    start = st.session_state.start

                # Mostrar resultados en pestañas
                result_tabs = st.tabs(
                    ["🎯 FIRST & FOLLOW", "📋 Tabla LL(1)", "🚀 Simulación", "🌳 Árbol de Derivación"]
                )

                # Pestaña FIRST & FOLLOW
                with result_tabs[0]:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Conjuntos FIRST")
                        for A in grammar:
                            st.markdown(f"**FIRST({A})** = {{ {', '.join(sorted(first[A]))} }}")
                    with col2:
                        st.subheader("Conjuntos FOLLOW")
                        for A in grammar:
                            st.markdown(f"**FOLLOW({A})** = {{ {', '.join(sorted(follow[A]))} }}")

                # Pestaña Tabla LL(1)
                with result_tabs[1]:
                    st.subheader("Tabla de Análisis LL(1)")
                    df = pd.DataFrame(index=grammar.keys(), columns=terminals)
                    for A in grammar:
                        for t in terminals:
                            prod = table[A].get(t)
                            df.loc[A, t] = " ".join(prod) if prod else ""
                    df.index.name = "No Terminal"

                    # Mejorar visualización de la tabla
                    st.markdown(
                        """
                    <div class="info-box">
                        Las celdas contienen las producciones a aplicar para cada combinación de no-terminal (filas) y token de entrada (columnas).
                        Las celdas vacías representan errores sintácticos.
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Estilizar la tabla
                    def highlight_nonempty(val):
                        if val != "":
                            return "background-color: #e6fff2"
                        return ""

                    st.dataframe(df.style.applymap(highlight_nonempty), use_container_width=True)

                # Pestaña Simulación y Árbol
                with result_tabs[2], result_tabs[3]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        token_input = st.text_input(
                            "Tokens separados por espacio:", placeholder="Ejemplo: id + id * id"
                        )
                    with col2:
                        simulate_button = st.button("▶️ Simular", type="primary", use_container_width=True)

                    if simulate_button and token_input:
                        tokens = token_input.split()

                        # Simulación (en pestaña 2)
                        with result_tabs[2]:
                            st.subheader("Traza de la Simulación")
                            trace = simulate_ll1(grammar, table, tokens, start)

                            # Colorear la tabla según el resultado
                            success = any(action == "Aceptado" for step in trace for action in step.values())

                            if success:
                                create_info_box("✅ La cadena de entrada ha sido aceptada por la gramática.", "success")
                            else:
                                create_info_box("❌ La cadena de entrada contiene errores sintácticos.", "warning")

                            # Mostrar la tabla de traza
                            trace_df = pd.DataFrame(trace)
                            st.dataframe(trace_df, use_container_width=True)

                        # Árbol (en pestaña 3)
                        with result_tabs[3]:
                            st.subheader("Árbol de Derivación")

                            try:
                                tree = build_parse_tree(grammar, table, tokens, start, empty_sym_input)
                                dot = tree_to_dot(tree)

                                # Leyenda para los colores
                                st.markdown(
                                    """
                                <div style="display: flex; gap: 20px; margin-bottom: 10px; font-size: 0.9rem;">
                                    <div><span style="background-color: lightblue; padding: 2px 8px; border-radius: 3px;">■</span> No-terminal</div>
                                    <div><span style="background-color: lightgreen; padding: 2px 8px; border-radius: 3px;">■</span> Terminal</div>
                                    <div><span style="background-color: lightyellow; padding: 2px 8px; border-radius: 3px;">■</span> Vacío (ε)</div>
                                    <div><span style="background-color: lightgrey; padding: 2px 8px; border-radius: 3px;">■</span> Fin ($)</div>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )

                                st.graphviz_chart(dot, use_container_width=True)
                            except Exception as e:
                                create_info_box(f"No se pudo generar el árbol: {str(e)}", "warning")

            except Exception as e:
                st.error(f"Error al procesar la gramática: {str(e)}")
                st.info("Verifica que tu gramática cumpla con el formato requerido.")


if __name__ == "__main__":
    main()
