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

        var hasMask = layer.mask && layer.mask.numProperties > 0;

        if (!hasMask) {
            // No mask - return layer info with noMask flag
            return JSON.stringify({
                name: layer.name,
                index: layer.index,
                width: layer.width,
                height: layer.height,
                noMask: true
            });
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
            noMask: false,
            numMasks: numMasks,
            selectedMaskIndex: selectedMaskIndex,
            selectedMaskName: layer.mask(selectedMaskIndex).name
        });
    } catch (e) {
        return JSON.stringify({ error: "getSelectedLayerWithMask: " + e.toString() });
    }
};

// Find a top-level project folder by name
AEI.findRootFolder = function(name) {
    var root = app.project.rootFolder;
    for (var i = 1; i <= root.numItems; i++) {
        var item = root.item(i);
        if (item instanceof FolderItem && item.name === name) return item;
    }
    return null;
};

// Remove temp solids' footage items (and AE's "Solids" folder if it was
// created just for them). Removing the temp comp leaves its solids behind —
// previously every inpaint left two solids in the project.
AEI.removeTempSolids = function(sources, solidsFolderExisted) {
    for (var i = 0; i < sources.length; i++) {
        try { if (sources[i]) sources[i].remove(); } catch (e) {}
    }
    if (!solidsFolderExisted) {
        var folder = AEI.findRootFolder("Solids");
        if (folder && folder.numItems === 0) {
            try { folder.remove(); } catch (e) {}
        }
    }
};

// Render the selected layer mask as PNG
AEI.renderLayerMask = function(layerIndex, maskIndex, outputPath) {
    var comp = app.project.activeItem;

    if (!comp || !(comp instanceof CompItem)) {
        return JSON.stringify({ error: "No active composition" });
    }

    var layer = comp.layer(layerIndex);
    if (!layer) {
        return JSON.stringify({ error: "Layer not found" });
    }

    var solidSources = [];
    var solidsFolderExisted = AEI.findRootFolder("Solids") !== null;
    var tempComp = null;

    try {
        tempComp = app.project.items.addComp(
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
        solidSources.push(blackSolid.source);

        // Solid must match source layer dimensions so mask coords are in the same space
        var whiteSolid = tempComp.layers.addSolid(
            [1, 1, 1],
            "_WhiteMask_",
            layer.width,
            layer.height,
            comp.pixelAspect
        );
        solidSources.push(whiteSolid.source);

        whiteSolid.position.setValue(layer.position.valueAtTime(comp.time, false));
        whiteSolid.anchorPoint.setValue(layer.anchorPoint.valueAtTime(comp.time, false));
        whiteSolid.scale.setValue(layer.scale.valueAtTime(comp.time, false));
        whiteSolid.rotation.setValue(layer.rotation.valueAtTime(comp.time, false));

        // Render ONLY the selected mask (maskIndex), using its own mode and
        // inverted flag. Previously this ignored maskIndex entirely and
        // rendered ALL masks on the layer merged together with a hardcoded
        // ADD mode — the "select one mask" behavior promised by the UI/README
        // never actually happened, and Subtract/Intersect/inverted masks
        // silently produced the wrong region.
        var numMasks = layer.mask.numProperties;
        if (maskIndex < 1 || maskIndex > numMasks) {
            maskIndex = 1;
        }
        var sourceMask = layer.mask(maskIndex);
        var newMask = whiteSolid.mask.addProperty("ADBE Mask Atom");

        newMask.maskPath.setValue(sourceMask.maskPath.valueAtTime(comp.time, false));
        newMask.maskFeather.setValue(sourceMask.maskFeather.valueAtTime(comp.time, false));
        newMask.maskExpansion.setValue(sourceMask.maskExpansion.valueAtTime(comp.time, false));
        // A single mask rendered alone should just be ADD (visible where the
        // path is) regardless of what mode it had among its siblings on the
        // source layer — Subtract/Intersect only mean something relative to
        // other masks being composited together, which we're not doing here.
        newMask.maskMode = MaskMode.ADD;
        // "Inverted" is a real, independent property though — preserve it,
        // or a mask the artist set to inverted renders as its own complement.
        newMask.inverted = sourceMask.inverted;

        tempComp.time = comp.time;

        var file = new File(outputPath);
        tempComp.saveFrameToPng(comp.time, file);

        tempComp.remove();
        AEI.removeTempSolids(solidSources, solidsFolderExisted);

        return JSON.stringify({ success: true, path: outputPath, maskIndex: maskIndex, maskName: sourceMask.name });

    } catch (e) {
        try { if (tempComp) tempComp.remove(); } catch (e2) {}
        AEI.removeTempSolids(solidSources, solidsFolderExisted);

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
        return JSON.stringify({ error: "Layer not found at index " + layerIndex });
    }

    if (!layer.source) {
        return JSON.stringify({ error: "Layer has no source" });
    }

    try {
        // Compute source time offset FIRST to size the temp comp correctly
        var sourceTime = comp.time - layer.startTime;
        if (sourceTime < 0) sourceTime = 0;

        // Temp comp must be long enough so that after startTime offset
        // the layer still covers render time 0
        var tempDuration = Math.max(1, sourceTime + 1);

        var tempComp = app.project.items.addComp(
            "_TempExport_" + layerIndex,
            comp.width,
            comp.height,
            comp.pixelAspect,
            tempDuration,
            comp.frameRate
        );

        // Add the layer's source footage
        var tempLayer = tempComp.layers.add(layer.source);

        // Match transform from original layer
        tempLayer.position.setValue(layer.position.valueAtTime(comp.time, false));
        tempLayer.anchorPoint.setValue(layer.anchorPoint.valueAtTime(comp.time, false));
        tempLayer.scale.setValue(layer.scale.valueAtTime(comp.time, false));
        tempLayer.rotation.setValue(layer.rotation.valueAtTime(comp.time, false));
        tempLayer.opacity.setValue([100]);

        // For video/sequence sources, offset so the right frame shows at time 0
        tempLayer.startTime = -sourceTime;

        // Render at time 0
        var file = new File(outputPath);
        tempComp.saveFrameToPng(0, file);

        // Cleanup
        tempComp.remove();

        return JSON.stringify({ success: true, path: outputPath });

    } catch (e) {
        // Cleanup on error
        try {
            for (var i = app.project.numItems; i >= 1; i--) {
                var item = app.project.item(i);
                if (item.name && item.name.indexOf("_TempExport_") === 0) {
                    item.remove();
                    break;
                }
            }
        } catch (e2) {}

        return JSON.stringify({ error: "Render failed: " + e.toString() });
    }
};

// Import PNG as new layer
// sourceLayerIndexOrName can be a number (index) or string (name)
// scalePercent: optional layer scale; upscale results are N× the comp size,
// so they're imported at 100/N % to keep the same framing at higher res.
AEI.importResultAsLayer = function(pngPath, sourceLayerIndexOrName, layerName, scalePercent) {
    var comp = app.project.activeItem;

    if (!comp || !(comp instanceof CompItem)) {
        return JSON.stringify({ error: "No active composition" });
    }

    var footage = null;
    var newLayer = null;
    try {
        var file = new File(pngPath);
        if (!file.exists) {
            return JSON.stringify({ error: "File not found: " + pngPath });
        }

        // Find source layer by index or name
        var sourceLayer;
        if (typeof sourceLayerIndexOrName === 'number') {
            sourceLayer = comp.layer(sourceLayerIndexOrName);
        } else {
            // Find by name
            for (var i = 1; i <= comp.numLayers; i++) {
                if (comp.layer(i).name === sourceLayerIndexOrName) {
                    sourceLayer = comp.layer(i);
                    break;
                }
            }
        }

        if (!sourceLayer) {
            return JSON.stringify({ error: "Source layer not found: " + sourceLayerIndexOrName });
        }

        // One undo step for the whole import
        app.beginUndoGroup("AE Inpaint: import result");

        var importOptions = new ImportOptions(file);
        footage = app.project.importFile(importOptions);

        // Keep results together instead of piling up in the project root
        var resultsFolder = AEI.findRootFolder("AE Inpaint Results");
        if (!resultsFolder) {
            resultsFolder = app.project.items.addFolder("AE Inpaint Results");
        }
        footage.parentFolder = resultsFolder;

        newLayer = comp.layers.add(footage);
        newLayer.name = layerName || "Inpaint Result";

        newLayer.moveBefore(sourceLayer);

        newLayer.startTime = sourceLayer.startTime;
        newLayer.inPoint = sourceLayer.inPoint;
        newLayer.outPoint = sourceLayer.outPoint;

        if (scalePercent && scalePercent !== 100) {
            newLayer.scale.setValue([scalePercent, scalePercent]);
        }

        // Result PNG is already rendered at comp dimensions with correct positioning
        // (renderLayerSolo bakes layer transform into the comp-sized output).
        // Do NOT copy source layer transform — it would apply it twice.
        // Default AE placement (center footage in comp) is correct for comp-sized footage.

        app.endUndoGroup();

        return JSON.stringify({
            success: true,
            layerName: newLayer.name,
            layerIndex: newLayer.index
        });

    } catch (e) {
        // Don't leave a half-imported result behind
        try { if (newLayer) newLayer.remove(); } catch (e2) {}
        try { if (footage) footage.remove(); } catch (e3) {}
        try { app.endUndoGroup(); } catch (e4) {}
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
    if (!folder.exists && !folder.create()) {
        return JSON.stringify({ error: "Can't create folder: " + outputFolder });
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

// Get selected layer (no mask required) - for upscale
AEI.getSelectedLayer = function() {
    try {
        var comp = app.project.activeItem;

        if (!comp || !(comp instanceof CompItem)) {
            return JSON.stringify({ error: "No active composition" });
        }

        if (comp.selectedLayers.length === 0) {
            return JSON.stringify({ error: "No layer selected" });
        }

        var layer = comp.selectedLayers[0];

        return JSON.stringify({
            name: layer.name,
            index: layer.index,
            width: layer.width,
            height: layer.height
        });
    } catch (e) {
        return JSON.stringify({ error: "getSelectedLayer: " + e.toString() });
    }
};

// Get all selected layers - for batch upscale
AEI.getSelectedLayers = function() {
    try {
        var comp = app.project.activeItem;

        if (!comp || !(comp instanceof CompItem)) {
            return JSON.stringify({ error: "No active composition" });
        }

        if (comp.selectedLayers.length === 0) {
            return JSON.stringify({ error: "No layers selected" });
        }

        var layers = [];
        for (var i = 0; i < comp.selectedLayers.length; i++) {
            var layer = comp.selectedLayers[i];
            layers.push({
                name: layer.name,
                index: layer.index,
                width: layer.width,
                height: layer.height,
                inPoint: layer.inPoint,
                outPoint: layer.outPoint,
                startTime: layer.startTime
            });
        }

        return JSON.stringify({
            layers: layers,
            count: layers.length
        });
    } catch (e) {
        return JSON.stringify({ error: "getSelectedLayers: " + e.toString() });
    }
};

// Export layer frame for upscale (no mask)
// Can accept layerIndex (number) or layerName (string)
AEI.exportLayerFrame = function(layerIndexOrName, outputFolder) {
    var comp = app.project.activeItem;

    if (!comp || !(comp instanceof CompItem)) {
        return JSON.stringify({ error: "No active composition" });
    }

    // Find layer by index or name
    var layer;
    var layerIndex;
    if (typeof layerIndexOrName === 'number') {
        layer = comp.layer(layerIndexOrName);
        layerIndex = layerIndexOrName;
    } else {
        // Find by name
        for (var i = 1; i <= comp.numLayers; i++) {
            if (comp.layer(i).name === layerIndexOrName) {
                layer = comp.layer(i);
                layerIndex = i;
                break;
            }
        }
    }

    if (!layer) {
        return JSON.stringify({ error: "Layer not found: " + layerIndexOrName });
    }

    var currentFrame = Math.round(comp.time * comp.frameRate);
    var layerName = layer.name.replace(/[^a-zA-Z0-9]/g, "_");
    var prefix = layerName + "_frame" + currentFrame + "_" + (new Date().getTime());

    var imagePath = outputFolder + "/" + prefix + ".png";

    var folder = new Folder(outputFolder);
    if (!folder.exists && !folder.create()) {
        return JSON.stringify({ error: "Can't create folder: " + outputFolder });
    }

    // Use current layer index (may have changed)
    var imageResult = JSON.parse(AEI.renderLayerSolo(layerIndex, imagePath));
    if (imageResult.error) {
        return JSON.stringify({ error: "Image export failed: " + imageResult.error });
    }

    return JSON.stringify({
        success: true,
        imagePath: imagePath,
        frame: currentFrame,
        compName: comp.name,
        layerName: layer.name,
        layerIndex: layerIndex
    });
};

// Create global aliases for easier calling
function getProjectInfo() { return $.global.AEInpaint.getProjectInfo(); }
function getSelectedLayerWithMask() { return $.global.AEInpaint.getSelectedLayerWithMask(); }
function getSelectedLayer() { return $.global.AEInpaint.getSelectedLayer(); }
function getSelectedLayers() { return $.global.AEInpaint.getSelectedLayers(); }
function renderLayerMask(a,b,c) { return $.global.AEInpaint.renderLayerMask(a,b,c); }
function renderLayerSolo(a,b) { return $.global.AEInpaint.renderLayerSolo(a,b); }
function importResultAsLayer(a,b,c,d) { return $.global.AEInpaint.importResultAsLayer(a,b,c,d); }
function exportForInpaint(a,b,c) { return $.global.AEInpaint.exportForInpaint(a,b,c); }
function exportLayerFrame(a,b) { return $.global.AEInpaint.exportLayerFrame(a,b); }
function testJSXLoaded() { return $.global.AEInpaint.testJSXLoaded(); }
