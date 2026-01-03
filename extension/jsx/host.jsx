/**
 * AE Inpaint - ExtendScript Host
 * Функции для работы с After Effects
 */

// Получить путь к папке проекта
function getProjectFolder() {
    var proj = app.project;
    if (!proj.file) {
        return JSON.stringify({ error: "Project not saved. Please save the project first." });
    }
    return JSON.stringify({ path: proj.file.parent.fsName });
}

// Получить информацию о текущем состоянии
function getProjectInfo() {
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
}

// Получить выбранный слой
function getSelectedLayer() {
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
        isAVLayer: layer instanceof AVLayer,
        isShapeLayer: layer instanceof ShapeLayer,
        isSolidLayer: layer instanceof AVLayer && layer.source instanceof SolidSource,
        hasMask: layer.mask ? layer.mask.numProperties > 0 : false,
        width: layer.width,
        height: layer.height
    });
}

// Найти слой-маску (Shape или Solid с белым цветом)
function findMaskLayer() {
    var comp = app.project.activeItem;

    if (!comp || !(comp instanceof CompItem)) {
        return JSON.stringify({ error: "No active composition" });
    }

    // Ищем слой с именем содержащим "mask" или "Mask"
    for (var i = 1; i <= comp.numLayers; i++) {
        var layer = comp.layer(i);
        var name = layer.name.toLowerCase();

        if (name.indexOf("mask") !== -1 || name.indexOf("inpaint") !== -1) {
            return JSON.stringify({
                found: true,
                name: layer.name,
                index: layer.index,
                isShapeLayer: layer instanceof ShapeLayer,
                isSolidLayer: layer instanceof AVLayer && layer.source instanceof SolidSource
            });
        }
    }

    // Если не нашли по имени, ищем ShapeLayer или белый Solid над выбранным слоем
    if (comp.selectedLayers.length > 0) {
        var selectedIndex = comp.selectedLayers[0].index;

        // Ищем слой выше выбранного
        if (selectedIndex > 1) {
            var aboveLayer = comp.layer(selectedIndex - 1);
            if (aboveLayer instanceof ShapeLayer ||
                (aboveLayer instanceof AVLayer && aboveLayer.source instanceof SolidSource)) {
                return JSON.stringify({
                    found: true,
                    name: aboveLayer.name,
                    index: aboveLayer.index,
                    isShapeLayer: aboveLayer instanceof ShapeLayer,
                    isSolidLayer: aboveLayer instanceof AVLayer && aboveLayer.source instanceof SolidSource
                });
            }
        }
    }

    return JSON.stringify({ found: false, error: "No mask layer found. Create a Shape or Solid layer named 'Mask'" });
}

// Рендер кадра в PNG
function renderFrameToPNG(layerIndex, outputPath) {
    var comp = app.project.activeItem;

    if (!comp || !(comp instanceof CompItem)) {
        return JSON.stringify({ error: "No active composition" });
    }

    var layer = comp.layer(layerIndex);
    if (!layer) {
        return JSON.stringify({ error: "Layer not found: " + layerIndex });
    }

    try {
        // Создаём временную композицию для рендера
        var tempComp = app.project.items.addComp(
            "TempRender",
            comp.width,
            comp.height,
            comp.pixelAspect,
            1 / comp.frameRate,
            comp.frameRate
        );

        // Дублируем слой в временную композицию
        var dupLayer = layer.duplicate();
        dupLayer.moveToComp(tempComp);
        dupLayer.startTime = 0;

        // Убираем слой из оригинальной композиции
        // (он уже перемещён)

        // Рендерим
        var file = new File(outputPath);
        tempComp.saveFrameToPng(comp.time, file);

        // Удаляем временную композицию
        tempComp.remove();

        return JSON.stringify({ success: true, path: outputPath });

    } catch (e) {
        return JSON.stringify({ error: "Render failed: " + e.toString() });
    }
}

// Альтернативный метод рендера через Output Module
function exportFramePNG(layerIndex, outputPath) {
    var comp = app.project.activeItem;

    if (!comp || !(comp instanceof CompItem)) {
        return JSON.stringify({ error: "No active composition" });
    }

    var currentFrame = Math.round(comp.time * comp.frameRate);

    try {
        // Создаём временную композицию только с нужным слоем
        var tempComp = app.project.items.addComp(
            "_TempExport_",
            comp.width,
            comp.height,
            comp.pixelAspect,
            1 / comp.frameRate,
            comp.frameRate
        );

        // Копируем слой
        var layer = comp.layer(layerIndex);
        layer.copyToComp(tempComp);

        var newLayer = tempComp.layer(1);
        newLayer.inPoint = 0;
        newLayer.outPoint = tempComp.duration;

        // Добавляем в очередь рендера
        var rqi = app.project.renderQueue.items.add(tempComp);
        var om = rqi.outputModule(1);

        // Настраиваем выход
        om.applyTemplate("_HIDDEN X-Factor 8 Premul");  // или другой PNG template
        om.file = new File(outputPath);

        // Рендерим
        app.project.renderQueue.render();

        // Удаляем временную композицию
        tempComp.remove();

        return JSON.stringify({ success: true, path: outputPath });

    } catch (e) {
        return JSON.stringify({ error: "Export failed: " + e.toString() });
    }
}

// Сохранить текущий кадр активной композиции
function saveCurrentFrame(outputPath) {
    var comp = app.project.activeItem;

    if (!comp || !(comp instanceof CompItem)) {
        return JSON.stringify({ error: "No active composition" });
    }

    try {
        var file = new File(outputPath);
        comp.saveFrameToPng(comp.time, file);
        return JSON.stringify({ success: true, path: outputPath });
    } catch (e) {
        return JSON.stringify({ error: "saveFrameToPng failed: " + e.toString() });
    }
}

// Рендер маски (с временным скрытием других слоёв)
function renderMaskToPNG(maskLayerIndex, outputPath) {
    var comp = app.project.activeItem;

    if (!comp || !(comp instanceof CompItem)) {
        return JSON.stringify({ error: "No active composition" });
    }

    try {
        // Запоминаем видимость всех слоёв
        var visibility = [];
        for (var i = 1; i <= comp.numLayers; i++) {
            visibility.push(comp.layer(i).enabled);
            comp.layer(i).enabled = (i === maskLayerIndex);
        }

        // Добавляем чёрный солид под маску
        var blackSolid = comp.layers.addSolid(
            [0, 0, 0],
            "_BlackBG_",
            comp.width,
            comp.height,
            comp.pixelAspect
        );
        blackSolid.moveToEnd();

        // Рендерим
        var file = new File(outputPath);
        comp.saveFrameToPng(comp.time, file);

        // Удаляем чёрный солид
        blackSolid.remove();

        // Восстанавливаем видимость
        for (var i = 1; i <= comp.numLayers; i++) {
            comp.layer(i).enabled = visibility[i - 1];
        }

        return JSON.stringify({ success: true, path: outputPath });

    } catch (e) {
        // Пытаемся восстановить видимость при ошибке
        try {
            for (var i = 1; i <= comp.numLayers; i++) {
                if (visibility && visibility[i - 1] !== undefined) {
                    comp.layer(i).enabled = visibility[i - 1];
                }
            }
        } catch (e2) {}

        return JSON.stringify({ error: "Mask render failed: " + e.toString() });
    }
}

// Импорт PNG как новый слой
function importResultAsLayer(pngPath, aboveLayerIndex, layerName) {
    var comp = app.project.activeItem;

    if (!comp || !(comp instanceof CompItem)) {
        return JSON.stringify({ error: "No active composition" });
    }

    try {
        // Импортируем файл
        var file = new File(pngPath);
        if (!file.exists) {
            return JSON.stringify({ error: "File not found: " + pngPath });
        }

        var importOptions = new ImportOptions(file);
        var footage = app.project.importFile(importOptions);

        // Добавляем в композицию
        var newLayer = comp.layers.add(footage);
        newLayer.name = layerName || "Inpaint Result";

        // Позиционируем над указанным слоем
        if (aboveLayerIndex && aboveLayerIndex > 0) {
            newLayer.moveBefore(comp.layer(aboveLayerIndex));
        }

        // Устанавливаем время на текущий кадр
        newLayer.startTime = comp.time;

        return JSON.stringify({
            success: true,
            layerName: newLayer.name,
            layerIndex: newLayer.index
        });

    } catch (e) {
        return JSON.stringify({ error: "Import failed: " + e.toString() });
    }
}

// Полный пайплайн экспорта для инпейнтинга
function exportForInpaint(sourceLayerIndex, maskLayerIndex, outputFolder) {
    var comp = app.project.activeItem;

    if (!comp || !(comp instanceof CompItem)) {
        return JSON.stringify({ error: "No active composition" });
    }

    var currentFrame = Math.round(comp.time * comp.frameRate);
    var prefix = comp.name.replace(/[^a-zA-Z0-9]/g, "_") + "_frame" + currentFrame;

    var imagePath = outputFolder + "/" + prefix + "_image.png";
    var maskPath = outputFolder + "/" + prefix + "_mask.png";

    // Создаём папку если не существует
    var folder = new Folder(outputFolder);
    if (!folder.exists) {
        folder.create();
    }

    // Рендерим исходный кадр
    var imageResult = JSON.parse(saveCurrentFrame(imagePath));
    if (imageResult.error) {
        return JSON.stringify({ error: "Image export failed: " + imageResult.error });
    }

    // Рендерим маску
    var maskResult = JSON.parse(renderMaskToPNG(maskLayerIndex, maskPath));
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
}
