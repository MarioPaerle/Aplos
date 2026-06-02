import dearpygui.dearpygui as dpg
import inspect
import sys
import os

# Aggiungiamo la root al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import everything available from Vathos
import Vathos._basics as basics
import Vathos.blocks as blocks

try:
    import Vathos.torch_layers.Attentions as attentions
except ImportError:
    attentions = None

dpg.create_context()

# --- REGISTRO DEL GRAFO (Per l'esportazione in Python) ---
NODE_REGISTRY = {}
LINK_REGISTRY = {}  # tracks connections safely
MODEL_IN_PIN = dpg.generate_uuid()
MODEL_OUT_PIN = dpg.generate_uuid()

# --- TEMI (UI STYLING) ---
with dpg.theme() as composition_link_theme:
    with dpg.theme_component(dpg.mvNodeLink):
        dpg.add_theme_color(dpg.mvNodeCol_Link, (255, 165, 0, 255))
        dpg.add_theme_color(dpg.mvNodeCol_LinkHovered, (255, 200, 50, 255))
        dpg.add_theme_color(dpg.mvNodeCol_LinkSelected, (255, 220, 100, 255))

with dpg.theme() as tensor_link_theme:
    with dpg.theme_component(dpg.mvNodeLink):
        dpg.add_theme_color(dpg.mvNodeCol_Link, (100, 200, 255, 255))
        dpg.add_theme_color(dpg.mvNodeCol_LinkHovered, (150, 220, 255, 255))
        dpg.add_theme_color(dpg.mvNodeCol_LinkSelected, (200, 240, 255, 255))

with dpg.theme() as io_node_theme:
    with dpg.theme_component(dpg.mvNode):
        dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, (0, 0, 0, 0))
        dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, (0, 0, 0, 0))
        dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, (0, 0, 0, 0))
        dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (0, 0, 0, 0))
        dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, (0, 0, 0, 0))
        dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, (0, 0, 0, 0))
        dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, (0, 0, 0, 0))
        dpg.add_theme_style(dpg.mvNodeStyleVar_NodePadding, 0, 0)
        try:
            dpg.add_theme_style(dpg.mvNodeStyleVar_NodeCornerRounding, 25)
        except AttributeError:
            pass


# --- 1. CARICAMENTO DINAMICO ---
def get_all_vathos_layers():
    all_classes = {}
    modules_to_scan = [basics, blocks]
    if attentions: modules_to_scan.append(attentions)

    for module in modules_to_scan:
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name in ('Layer', 'Builder', 'Renamer', 'VathosConfig', 'Module'): continue
            if issubclass(obj, basics.nn.Module) or getattr(obj, '_is_vathos_layer', False):
                all_classes[name] = obj
    return all_classes


ALL_BLOCKS = get_all_vathos_layers()


# --- 2. INTROSPEZIONE ---
def get_class_params(cls):
    try:
        sig = inspect.signature(cls.__init__)
        params = {'static': {}, 'slots': []}
        for name, param in sig.parameters.items():
            if name in ('self', 'args', 'kwargs'): continue

            ann = str(param.annotation).lower()
            if 'layer' in ann or 'module' in ann or 'callable' in ann or 'mixer' in name or name == 'embedder' or name == 'unembedder':
                params['slots'].append(name)
                continue

            default = param.default if param.default is not inspect.Parameter.empty else None
            param_type = type(default) if default is not None else str
            if param.annotation is not inspect.Parameter.empty and param_type is str:
                if param.annotation == int:
                    param_type = int
                elif param.annotation == float:
                    param_type = float
                elif param.annotation == bool:
                    param_type = bool

            params['static'][name] = {'default': default, 'type': param_type}
        return params
    except ValueError:
        return {'static': {}, 'slots': []}


# --- 3. LOGICA DI ESPORTAZIONE IN CODICE PYTHON ---
def generate_vathos_code():
    links = dpg.get_item_children("NodeEditor", 1) or []

    # Mappatura Pin -> Nodo
    pin_to_node = {}
    for nid, ndata in NODE_REGISTRY.items():
        pin_to_node[ndata['tensor_in']] = (nid, 'tensor_in', None)
        pin_to_node[ndata['tensor_out']] = (nid, 'tensor_out', None)
        for slot_name, pin_id in ndata['slots'].items():
            pin_to_node[pin_id] = (nid, 'slot', slot_name)

    pin_to_node[MODEL_IN_PIN] = ('MODEL_IN', 'tensor_out', None)
    pin_to_node[MODEL_OUT_PIN] = ('MODEL_OUT', 'tensor_in', None)

    slot_deps = {nid: {} for nid in NODE_REGISTRY}
    tensor_flow = {}

    # Parse connections through our safe registry
    for link in links:
        if link not in LINK_REGISTRY:
            continue

        src, dst = LINK_REGISTRY[link]
        if dst in pin_to_node and src in pin_to_node:
            dst_node, dst_type, dst_slot = pin_to_node[dst]
            src_node, src_type, _ = pin_to_node[src]

            if dst_type == 'slot':
                slot_deps[dst_node][dst_slot] = src_node
            elif dst_type == 'tensor_in':
                tensor_flow[dst_node] = src_node

    # Topological sort for __init__ (resolve inner dependencies first)
    init_order = []
    visited = set()

    def visit(nid):
        if nid in visited or nid == 'MODEL_IN' or nid == 'MODEL_OUT': return
        for dep in slot_deps.get(nid, {}).values():
            visit(dep)
        visited.add(nid)
        init_order.append(nid)

    for nid in NODE_REGISTRY: visit(nid)

    # GENERAZIONE TESTO CODICE
    code = "import torch\nimport torch.nn as nn\n"
    code += "from Vathos.blocks import *\nfrom Vathos._basics import *\n\n"
    code += "class GeneratedVathosModel(nn.Module):\n"
    code += "    def __init__(self):\n"
    code += "        super().__init__()\n"

    var_names = {}
    # Scrittura dell' __init__
    for i, nid in enumerate(init_order):
        ndata = NODE_REGISTRY[nid]
        cname = ndata['class_name']
        var_name = f"{cname.lower()}_{i}"
        var_names[nid] = var_name

        kwargs = []
        for pname, ptag in ndata['static_inputs'].items():
            val = dpg.get_value(ptag)
            if isinstance(val, str):
                kwargs.append(f"{pname}='{val}'")
            else:
                kwargs.append(f"{pname}={val}")

        for sname, dep_id in slot_deps[nid].items():
            kwargs.append(f"{sname}=self.{var_names.get(dep_id, 'None')}")

        args_str = ", ".join(kwargs)

        repeat = dpg.get_value(ndata['repeat_input'])
        if repeat > 1:
            code += f"        self.{var_name} = BlockStack([{cname}({args_str}) for _ in range({repeat})])\n"
        else:
            code += f"        self.{var_name} = {cname}({args_str})\n"

    # Scrittura del forward pass
    code += "\n    def forward(self, x):\n"
    current = 'MODEL_IN'
    forward_order = []

    def get_next(node):
        for dest, src in tensor_flow.items():
            if src == node: return dest
        return None

    nxt = get_next(current)
    while nxt and nxt != 'MODEL_OUT':
        forward_order.append(nxt)
        nxt = get_next(nxt)

    if not forward_order:
        code += "        # Nessun flusso tensoriale collegato all'input!\n"
        code += "        return x\n"
    else:
        for nid in forward_order:
            code += f"        x = self.{var_names[nid]}(x)\n"
        code += "        return x\n"

    code += "\n# Instantiate and test the model\nif __name__ == '__main__':\n"
    code += "    model = GeneratedVathosModel()\n"
    code += "    print(model)\n"
    return code


def show_export_window(code):
    if dpg.does_alias_exist("export_window"):
        dpg.delete_item("export_window")

    with dpg.window(label="Codice Vathos Generato", tag="export_window", width=800, height=600, pos=[300, 100]):
        dpg.add_input_text(default_value=code, multiline=True, width=-1, height=-40, readonly=True)
        dpg.add_button(label="Chiudi", callback=lambda: dpg.delete_item("export_window"), width=-1)


def export_callback(sender, app_data, user_data):
    if user_data == "vathos_code":
        code = generate_vathos_code()
        show_export_window(code)


# --- 4. RICERCA E UI NODI ---
def search_callback(sender, app_data):
    search_str = app_data.lower()
    for name in ALL_BLOCKS.keys():
        btn_tag = f"menu_btn_{name}"
        if search_str in name.lower():
            dpg.show_item(btn_tag)
        else:
            dpg.hide_item(btn_tag)


spawn_x, spawn_y = 300, 100


def spawn_node_from_menu(sender, app_data, user_data):
    global spawn_x, spawn_y
    spawn_node(user_data, pos=[spawn_x, spawn_y])
    spawn_x += 30
    spawn_y += 30
    if spawn_x > 600:
        spawn_x, spawn_y = 300, 100


def spawn_node(class_name, pos=None):
    if class_name not in ALL_BLOCKS: return
    cls = ALL_BLOCKS[class_name]
    params = get_class_params(cls)

    # Generazione ID per il tracciamento logico
    node_id = dpg.generate_uuid()
    tensor_in_pin = dpg.generate_uuid()
    tensor_out_pin = dpg.generate_uuid()
    repeat_pin = dpg.generate_uuid()

    ndata = {
        'class_name': class_name,
        'static_inputs': {},
        'slots': {},
        'tensor_in': tensor_in_pin,
        'tensor_out': tensor_out_pin,
        'repeat_input': repeat_pin
    }
    NODE_REGISTRY[node_id] = ndata

    with dpg.node(parent="NodeEditor", label=class_name, pos=pos or [300, 100], tag=node_id):
        # Pin Input: Flusso Dati
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, user_data="tensor", tag=tensor_in_pin):
            dpg.add_text("Dati In", color=[100, 200, 255])

        # Pin Input: Componenti Architetturali
        for slot in params['slots']:
            slot_pin = dpg.generate_uuid()
            ndata['slots'][slot] = slot_pin
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, user_data="composition", tag=slot_pin):
                dpg.add_text(f" Slot: {slot} ", color=[255, 165, 0])

        # Parametri statici
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_input_int(label="Repeat (xN)", default_value=1, width=80, tag=repeat_pin)
            dpg.add_separator()
            for param_name, info in params['static'].items():
                p_type, p_def = info['type'], info['default']
                ptag = dpg.generate_uuid()
                ndata['static_inputs'][param_name] = ptag

                if p_type == int:
                    dpg.add_input_int(label=param_name, default_value=p_def or 0, width=120, tag=ptag)
                elif p_type == float:
                    dpg.add_input_float(label=param_name, default_value=p_def or 0.0, width=120, tag=ptag)
                elif p_type == bool:
                    dpg.add_checkbox(label=param_name, default_value=bool(p_def), tag=ptag)
                else:
                    dpg.add_input_text(label=param_name, default_value=str(p_def or ""), width=120, tag=ptag)

        # Pin Output
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, user_data="tensor", tag=tensor_out_pin):
            dpg.add_text("Dati Out", color=[100, 200, 255])


# --- 5. CALLBACKS DEI LINK ---
def link_callback(sender, app_data):
    source_pin, dest_pin = app_data[0], app_data[1]

    link_id = dpg.add_node_link(source_pin, dest_pin, parent=sender)
    # Save the connection in our registry
    LINK_REGISTRY[link_id] = (source_pin, dest_pin)

    pin_type = dpg.get_item_user_data(dest_pin)
    if pin_type == "composition":
        dpg.bind_item_theme(link_id, composition_link_theme)
    else:
        dpg.bind_item_theme(link_id, tensor_link_theme)


def delink_callback(sender, app_data):
    dpg.delete_item(app_data)
    # Remove the link from our registry
    if app_data in LINK_REGISTRY:
        del LINK_REGISTRY[app_data]


# --- 6. SETUP FINESTRA PRINCIPALE ---
with dpg.window(tag="MainWindow"):
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Nuovo")
        with dpg.menu(label="Esporta"):
            dpg.add_menu_item(label="Codice Vathos (.py)", callback=export_callback, user_data="vathos_code")

    with dpg.group(horizontal=True):
        # SIDEBAR
        with dpg.child_window(width=300, border=True):
            dpg.add_text("COMPONENTI VATHOS", color=[100, 255, 150])
            dpg.add_input_text(hint="Cerca blocco...", width=-1, callback=search_callback)
            dpg.add_separator()

            with dpg.child_window(width=-1, height=-160, border=False):
                for name in sorted(ALL_BLOCKS.keys()):
                    dpg.add_button(label=f" + {name}", width=-1, callback=spawn_node_from_menu,
                                   user_data=name, tag=f"menu_btn_{name}")

            # Istruzioni
            with dpg.child_window(height=150, border=True):
                dpg.add_text("CONTROLLI:", color=[100, 200, 255])
                dpg.add_text("• Zoom: CTRL + Rotellina", color=[255, 255, 200])
                dpg.add_text("• Pan: Tasto Centrale", color=[150, 150, 150])
                dpg.add_separator()
                dpg.add_text("• Arancione: Mixer", color=[255, 165, 0])
                dpg.add_text("• Azzurro: Tensore Dati", color=[100, 200, 255])

        # NODE EDITOR
        with dpg.node_editor(
                tag="NodeEditor",
                callback=link_callback,
                delink_callback=delink_callback,
                minimap=True,
                minimap_location=3
        ):
            # INPUT FISSO
            with dpg.node(label="", pos=[100, 350], tag="node_model_in", draggable=False):
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, user_data="tensor", tag=MODEL_IN_PIN):
                    dpg.add_text(" (x) ", color=[150, 150, 150])
            dpg.bind_item_theme("node_model_in", io_node_theme)

            # OUTPUT FISSO
            with dpg.node(label="", pos=[1000, 350], tag="node_model_out", draggable=False):
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, user_data="tensor", tag=MODEL_OUT_PIN):
                    dpg.add_text(" (ret) ", color=[150, 150, 150])
            dpg.bind_item_theme("node_model_out", io_node_theme)

dpg.create_viewport(title='Vathos Visual Builder - Esportatore Python', width=1400, height=800)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("MainWindow", True)
dpg.start_dearpygui()
dpg.destroy_context()