const { app } = window.comfyAPI.app;

const LOCALIZATION_URL = "/easy_sensenova_u1/local/zh-cn/nodes.json";
let localizationPromise = null;

function loadLocalization() {
  if (!localizationPromise) {
    localizationPromise = fetch(LOCALIZATION_URL)
      .then((response) => (response.ok ? response.json() : {}))
      .catch((error) => {
        console.warn("[Comfy-Easy-SenseNova-U1] 加载简体中文翻译失败:", error);
        return {};
      });
  }
  return localizationPromise;
}

function translateNodeData(nodeData, translations) {
  const translated = translations?.nodes?.[nodeData?.name];
  if (!translated) return;
  if (translated.display_name) nodeData.display_name = translated.display_name;

  if (Array.isArray(nodeData.output_name)) {
    nodeData.output_name = nodeData.output_name.map(
      (name) => translated.outputs?.[name]?.display_name ?? translated.outputs?.[name] ?? name,
    );
  }

  for (const section of ["required", "optional", "hidden"]) {
    const inputs = nodeData.input?.[section];
    if (!inputs) continue;
    for (const [name, spec] of Object.entries(inputs)) {
      const entry = translated.inputs?.[name] ?? translations?.input_labels?.[name];
      if (!entry || !Array.isArray(spec)) continue;
      const options = spec[1] ?? {};
      if (entry.display_name) {
        options.display_name = entry.display_name;
        options.label = entry.display_name;
      }
      if (entry.tooltip) options.tooltip = entry.tooltip;
      spec[1] = options;
    }
  }
}

function translateInstance(node, translations) {
  const nodeClass = node.constructor?.comfyClass ?? node.type;
  const translated = translations?.nodes?.[nodeClass];
  if (!translated) return;
  if (translated.display_name) node.title = translated.display_name;
  for (const slot of node.inputs ?? []) {
    const entry = translated.inputs?.[slot.name] ?? translations?.input_labels?.[slot.name];
    if (entry?.display_name) slot.label = slot.localized_name = entry.display_name;
  }
  for (const widget of node.widgets ?? []) {
    const entry = translated.inputs?.[widget.name] ?? translations?.input_labels?.[widget.name];
    if (entry?.display_name) widget.label = widget.localized_name = entry.display_name;
  }
  for (const slot of node.outputs ?? []) {
    const entry = translated.outputs?.[slot.name] ?? translated.outputs?.[slot.label];
    const label = entry?.display_name ?? entry;
    if (label) slot.label = slot.localized_name = label;
  }
}

function chainCallback(target, name, callback) {
  const original = target[name];
  target[name] = function (...args) {
    const result = original?.apply(this, args);
    callback.apply(this, args);
    return result;
  };
}

app.registerExtension({
  name: "eastmoe.ComfyEasySenseNovaU1.i18n",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const translations = await loadLocalization();
    translateNodeData(nodeData, translations);
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      translateInstance(this, translations);
    });
    chainCallback(nodeType.prototype, "onConfigure", function () {
      translateInstance(this, translations);
    });
  },
  async nodeCreated(node) {
    translateInstance(node, await loadLocalization());
  },
});
