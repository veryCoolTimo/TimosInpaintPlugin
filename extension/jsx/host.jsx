/**
 * AE Inpaint - ExtendScript Host
 * Functions stored in $.global to persist between evalScript calls
 */

// JSON polyfill for ExtendScript
if (typeof JSON === 'undefined') {
    JSON = {
        stringify: function(obj) {
            if (obj === null) return 'null';
            if (typeof obj === 'undefined') return 'undefined';
            if (typeof obj === 'number' || typeof obj === 'boolean') return String(obj);
            if (typeof obj === 'string') return '"' + obj.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n').replace(/\r/g, '\\r') + '"';
            if (obj instanceof Array) {
                var arr = [];
                for (var i = 0; i < obj.length; i++) {
                    arr.push(JSON.stringify(obj[i]));
                }
                return '[' + arr.join(',') + ']';
            }
            if (typeof obj === 'object') {
                var parts = [];
                for (var key in obj) {
                    if (obj.hasOwnProperty(key)) {
                        parts.push('"' + key + '":' + JSON.stringify(obj[key]));
                    }
                }
                return '{' + parts.join(',') + '}';
            }
            return '{}';
        },
        parse: function(str) {
            return eval('(' + str + ')');
        }
    };
}

// Initialize global namespace
if (typeof $.global.AEInpaint === 'undefined') {
    $.global.AEInpaint = {};
}

var AEI = $.global.AEInpaint;

// Get project info
AEI.getProjectInfo = function() {
    try {
        var proj = app.project;
        var comp = proj.activeItem;

        if (!comp || !(comp instanceof CompItem)) {
            return JSON.stringify({ error: "No active composition" });
        }

        return JSON.stringify({
            projectPath: proj.file ? proj.file.parent.fsName : null,
            compName: comp.name,
            compWidth: comp.width,
            compHeight: comp.height,
            currentTime: comp.time,
            frameRate: comp.frameRate,
            currentFrame: Math.round(comp.time * comp.frameRate)
        });
    } catch (e) {
        return JSON.stringify({ error: "getProjectInfo: " + e.toString() });
    }
};

// Get selected layer with mask info
AEI.getSelectedLayerWithMask = function() {
    try {
        var comp = app.project.activeItem;

        if (!comp || !(comp instanceof CompItem)) {
            return JSON.stringify({ error: "No active composition" });
        }

        if (comp.selectedLayers.length === 0) {
            return JSON.stringify({ error: "No layer selected" });
        }

        var layer = comp.selectedLayers[0];

        if (!layer.mask || layer.mask.numProperties === 0) {
            return JSON.stringify({ error: "No mask on selected layer. Draw a mask first (Pen tool)." });
        }

        var numMasks = layer.mask.numProperties;
        var selectedMaskIndex = 1;

        for (var i = 1; i <= numMasks; i++) {
            var mask = layer.mask(i);
            if (mask.selected) {
                selectedMaskIndex = i;
                break;
            }
        }

        return JSON.stringify({
            name: layer.name,
            index: layer.index,
            width: layer.width,
            height: layer.height,
            numMasks: numMasks,
            selectedMaskIndex: selectedMaskIndex,
            selectedMaskName: layer.mask(selectedMaskIndex).name
        });
    } catch (e) {
        return JSON.stringify({ error: "getSelectedLayerWithMask: " + e.toString() });
    }
};

// Render ALL layer masks as PNG (combined)
AEI.renderLayerMask = function(layerIndex, maskIndex, outputPath) {
    var comp = app.project.activeItem;

    if (!comp || !(comp instanceof CompItem)) {
        return JSON.stringify({ error: "No active composition" });
    }

    var layer = comp.layer(layerIndex);
    if (!layer) {
        return JSON.stringify({ error: "Layer not found" });
    }

    try {
        var tempComp = app.project.items.addComp(
            "_MaskRender_",
            comp.width,
            comp.height,
            comp.pixelAspect,
            comp.duration,
            comp.frameRate
        );

        var blackSolid = tempComp.layers.addSolid(
            [0, 0, 0],
            "_BlackBG_",
            comp.width,
            comp.height,
            comp.pixelAspect
        );

        var whiteSolid = tempComp.layers.addSolid(
            [1, 1, 1],
            "_WhiteMask_",
            comp.width,
            comp.height,
            comp.pixelAspect
        );

        whiteSolid.position.setValue(layer.position.valueAtTime(comp.time, false));
        whiteSolid.anchorPoint.setValue(layer.anchorPoint.valueAtTime(comp.time, false));
        whiteSolid.scale.setValue(layer.scale.valueAtTime(comp.time, false));
        whiteSolid.rotation.setValue(layer.rotation.valueAtTime(comp.time, false));

        // Add ALL masks from the layer (not just selected one)
        var numMasks = layer.mask.numProperties;
        for (var i = 1; i <= numMasks; i++) {
            var sourceMask = layer.mask(i);
            var newMask = whiteSolid.mask.addProperty("ADBE Mask Atom");

            newMask.maskPath.setValue(sourceMask.maskPath.valueAtTime(comp.time, false));
            // Use feather and expansion from AE mask properties
            newMask.maskFeather.setValue(sourceMask.maskFeather.valueAtTime(comp.time, false));
            newMask.maskExpansion.setValue(sourceMask.maskExpansion.valueAtTime(comp.time, false));
            newMask.maskMode = MaskMode.ADD;
        }

        tempComp.time = comp.time;

        var file = new File(outputPath);
        tempComp.saveFrameToPng(comp.time, file);

        tempComp.remove();

        return JSON.stringify({ success: true, path: outputPath, masksUsed: numMasks });

    } catch (e) {
        try {
            for (var i = app.project.numItems; i >= 1; i--) {
                if (app.project.item(i).name === "_MaskRender_") {
                    app.project.item(i).remove();
                    break;
                }
            }
        } catch (e2) {}

        return JSON.stringify({ error: "Mask render failed: " + e.toString() });
    }
};

// Render layer solo as PNG
AEI.renderLayerSolo = function(layerIndex, outputPath) {
    var comp = app.project.activeItem;

    if (!comp || !(comp instanceof CompItem)) {
        return JSON.stringify({ error: "No active composition" });
    }

    var layer = comp.layer(layerIndex);
    if (!layer) {
        return JSON.stringify({ error: "Layer not found" });
    }

    try {
        var maskStates = [];
        if (layer.mask) {
            for (var i = 1; i <= layer.mask.numProperties; i++) {
                maskStates.push(layer.mask(i).maskMode);
                layer.mask(i).maskMode = MaskMode.NONE;
            }
        }

        var visibility = [];
        for (var i = 1; i <= comp.numLayers; i++) {
            visibility.push(comp.layer(i).enabled);
            comp.layer(i).enabled = (i === layerIndex);
        }

        var file = new File(outputPath);
        comp.saveFrameToPng(comp.time, file);

        for (var i = 1; i <= comp.numLayers; i++) {
            comp.layer(i).enabled = visibility[i - 1];
        }

        if (layer.mask) {
            for (var i = 1; i <= layer.mask.numProperties; i++) {
                layer.mask(i).maskMode = maskStates[i - 1];
            }
        }

        return JSON.stringify({ success: true, path: outputPath });

    } catch (e) {
        return JSON.stringify({ error: "Layer render failed: " + e.toString() });
    }
};

// Import PNG as new layer
AEI.importResultAsLayer = function(pngPath, sourceLayerIndex, layerName) {
    var comp = app.project.activeItem;

    if (!comp || !(comp instanceof CompItem)) {
        return JSON.stringify({ error: "No active composition" });
    }

    try {
        var file = new File(pngPath);
        if (!file.exists) {
            return JSON.stringify({ error: "File not found: " + pngPath });
        }

        var importOptions = new ImportOptions(file);
        var footage = app.project.importFile(importOptions);

        var sourceLayer = comp.layer(sourceLayerIndex);
        var newLayer = comp.layers.add(footage);
        newLayer.name = layerName || "Inpaint Result";

        newLayer.moveBefore(sourceLayer);

        newLayer.startTime = sourceLayer.startTime;
        newLayer.inPoint = sourceLayer.inPoint;
        newLayer.outPoint = sourceLayer.outPoint;

        newLayer.position.setValue(sourceLayer.position.valueAtTime(comp.time, false));
        newLayer.anchorPoint.setValue(sourceLayer.anchorPoint.valueAtTime(comp.time, false));
        newLayer.scale.setValue(sourceLayer.scale.valueAtTime(comp.time, false));
        newLayer.rotation.setValue(sourceLayer.rotation.valueAtTime(comp.time, false));
        newLayer.opacity.setValue(sourceLayer.opacity.valueAtTime(comp.time, false));

        return JSON.stringify({
            success: true,
            layerName: newLayer.name,
            layerIndex: newLayer.index
        });

    } catch (e) {
        return JSON.stringify({ error: "Import failed: " + e.toString() });
    }
};

// Export for inpainting
AEI.exportForInpaint = function(layerIndex, maskIndex, outputFolder) {
    var comp = app.project.activeItem;

    if (!comp || !(comp instanceof CompItem)) {
        return JSON.stringify({ error: "No active composition" });
    }

    var currentFrame = Math.round(comp.time * comp.frameRate);
    var prefix = comp.name.replace(/[^a-zA-Z0-9]/g, "_") + "_frame" + currentFrame;

    var imagePath = outputFolder + "/" + prefix + "_image.png";
    var maskPath = outputFolder + "/" + prefix + "_mask.png";

    var folder = new Folder(outputFolder);
    if (!folder.exists) {
        folder.create();
    }

    var imageResult = JSON.parse(AEI.renderLayerSolo(layerIndex, imagePath));
    if (imageResult.error) {
        return JSON.stringify({ error: "Image export failed: " + imageResult.error });
    }

    var maskResult = JSON.parse(AEI.renderLayerMask(layerIndex, maskIndex, maskPath));
    if (maskResult.error) {
        return JSON.stringify({ error: "Mask export failed: " + maskResult.error });
    }

    return JSON.stringify({
        success: true,
        imagePath: imagePath,
        maskPath: maskPath,
        frame: currentFrame,
        compName: comp.name
    });
};

// Test function
AEI.testJSXLoaded = function() {
    return JSON.stringify({ loaded: true, version: "1.0" });
};

// Create global aliases for easier calling
function getProjectInfo() { return $.global.AEInpaint.getProjectInfo(); }
function getSelectedLayerWithMask() { return $.global.AEInpaint.getSelectedLayerWithMask(); }
function renderLayerMask(a,b,c) { return $.global.AEInpaint.renderLayerMask(a,b,c); }
function renderLayerSolo(a,b) { return $.global.AEInpaint.renderLayerSolo(a,b); }
function importResultAsLayer(a,b,c) { return $.global.AEInpaint.importResultAsLayer(a,b,c); }
function exportForInpaint(a,b,c) { return $.global.AEInpaint.exportForInpaint(a,b,c); }
function testJSXLoaded() { return $.global.AEInpaint.testJSXLoaded(); }
