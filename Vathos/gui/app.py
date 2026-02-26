import dearpygui.dearpygui as dpg
import inspect
import sys
import os
import json

# Aggiungiamo la root al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importiamo tutto il possibile da Vathos
import Vathos._basics as basics
import Vathos.blocks as blocks

try:
    import Vathos.torch_layers.Attentions as attentions
except ImportError:
    attentions = None

dpg.create_context()

# --- TEMI (UI STYLING) ---

# 1. Tema per i cavi di Composizione (Arancioni)
with dpg.theme() as composition_link_theme:
    with dpg.theme_component(dpg.mvNodeLink):
        dpg.add_theme_color(dpg.mvNodeCol_Link, (255, 165, 0, 255))
        dpg.add_theme_color(dpg.mvNodeCol_LinkHovered, (255, 200, 50, 255))
        dpg.add_theme_color(dpg.mvNodeCol_LinkSelected, (255, 220, 100, 255))

# 2. Tema per i cavi Tensor / Dati (Azzurrini)
with dpg.theme() as tensor_link_theme:
    with dpg.theme_component(dpg.mvNodeLink):
        dpg.add_theme_color(dpg.mvNodeCol_Link, (100, 200, 255, 255))
        dpg.add_theme_color(dpg.mvNodeCol_LinkHovered, (150, 220, 255, 255))
        dpg.add_theme_color(dpg.mvNodeCol_LinkSelected, (200, 240, 255, 255))

# 3. Tema per Nodi Input/Output (Semplici cerchi grigi minimalisti)
with dpg.theme() as io_node_theme:
    with dpg.theme_component(dpg.mvNode):
        # Nascondiamo tutto tranne il pin
        dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, (0, 0, 0, 0))
        dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, (0, 0, 0, 0))
        dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, (0, 0, 0, 0))
        dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (0, 0, 0, 0))
        dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, (0, 0, 0, 0))
        dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, (0, 0, 0, 0))
        dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, (0, 0, 0, 0))
        # Riduciamo il padding per farlo sembrare un cerchio piccolo
        dpg.add_theme_style(dpg.mvNodeStyleVar_NodePadding, 0, 0)
        try:
            dpg.add_theme_style(dpg.mvNodeStyleVar_NodeCornerRounding, 25)
        except AttributeError:
            pass


# --- 1. CARICAMENTO DINAMICO DI TUTTI I BLOCCHI ---
def get_all_vathos_layers():
    """Scansiona i moduli per trovare tutte le classi Vathos o PyTorch."""
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
    """Estrae i parametri, separando statici dagli Slots di composizione."""
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


# --- 3. RICERCA E UI ---
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

    with dpg.node(parent="NodeEditor", label=class_name, pos=pos or [300, 100]):
        # Pin Input: Flusso Dati (Tensor)
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, user_data="tensor"):
            dpg.add_text("Dati In", color=[100, 200, 255])

        # Pin Input: Componenti Architetturali (Mixers)
        for slot in params['slots']:
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, user_data="composition"):
                dpg.add_text(f" Slot: {slot} ", color=[255, 165, 0])

        # Parametri statici
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_input_int(label="Repeat (xN)", default_value=1, width=80)
            dpg.add_separator()
            for param_name, info in params['static'].items():
                p_type, p_def = info['type'], info['default']
                if p_type == int:
                    dpg.add_input_int(label=param_name, default_value=p_def or 0, width=120)
                elif p_type == float:
                    dpg.add_input_float(label=param_name, default_value=p_def or 0.0, width=120)
                elif p_type == bool:
                    dpg.add_checkbox(label=param_name, default_value=bool(p_def))
                else:
                    dpg.add_input_text(label=param_name, default_value=str(p_def or ""), width=120)

        # Pin Output
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, user_data="tensor"):
            dpg.add_text("Dati Out", color=[100, 200, 255])


# --- 4. CALLBACKS DEI LINK ---
def link_callback(sender, app_data):
    source_pin, dest_pin = app_data[0], app_data[1]

    # Crea il link
    link_id = dpg.add_node_link(source_pin, dest_pin, parent=sender)

    # Recupera il tipo di pin di destinazione per decidere il colore
    pin_type = dpg.get_item_user_data(dest_pin)

    if pin_type == "composition":
        dpg.bind_item_theme(link_id, composition_link_theme)
    else:
        dpg.bind_item_theme(link_id, tensor_link_theme)


def delink_callback(sender, app_data):
    dpg.delete_item(app_data)


# --- 5. SETUP FINESTRA ---
with dpg.window(tag="MainWindow"):
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Nuovo")
        with dpg.menu(label="Esporta"):
            dpg.add_menu_item(label="Codice Vathos (.py)")

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

            # Istruzioni aggiornate
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
            # INPUT FISSO (Pallino grigio discreto)
            with dpg.node(label="", pos=[100, 350], tag="node_model_in", draggable=False):
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, user_data="tensor"):
                    dpg.add_text(" (x) ", color=[150, 150, 150])
            dpg.bind_item_theme("node_model_in", io_node_theme)

            # OUTPUT FISSO (Pallino grigio discreto)
            with dpg.node(label="", pos=[1000, 350], tag="node_model_out", draggable=False):
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, user_data="tensor"):
                    dpg.add_text(" (ret) ", color=[150, 150, 150])
            dpg.bind_item_theme("node_model_out", io_node_theme)

# Configurazione finale del Viewport
dpg.create_viewport(title='Vathos Visual Builder', width=1400, height=800)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("MainWindow", True)
dpg.start_dearpygui()
dpg.destroy_context()