/**
 * CSInterface - Adobe CEP Communication Interface
 * Минимальная версия для AE Inpaint
 */

function CSInterface() {}

/**
 * Версия API
 */
CSInterface.prototype.CYCLIC = "cyclic";

/**
 * Выполнение ExtendScript
 */
CSInterface.prototype.evalScript = function(script, callback) {
    if (typeof window.__adobe_cep__ !== "undefined") {
        window.__adobe_cep__.evalScript(script, callback);
    } else {
        // Fallback для тестирования без AE
        console.log("evalScript:", script);
        if (callback) {
            callback('{"error": "Not running in Adobe environment"}');
        }
    }
};

/**
 * Получение системного пути
 */
CSInterface.prototype.getSystemPath = function(pathType) {
    var path = "";
    if (typeof window.__adobe_cep__ !== "undefined") {
        path = window.__adobe_cep__.getSystemPath(pathType);
    }
    return path;
};

/**
 * Типы системных путей
 */
CSInterface.prototype.EXTENSION = "extension";
CSInterface.prototype.USER_DATA = "userData";
CSInterface.prototype.COMMON_FILES = "commonFiles";
CSInterface.prototype.MY_DOCUMENTS = "myDocuments";
CSInterface.prototype.HOST_APPLICATION = "hostApplication";

/**
 * Получение информации о хост-приложении
 */
CSInterface.prototype.getHostEnvironment = function() {
    if (typeof window.__adobe_cep__ !== "undefined") {
        return JSON.parse(window.__adobe_cep__.getHostEnvironment());
    }
    return {
        appName: "Unknown",
        appVersion: "0.0.0"
    };
};

/**
 * Закрытие расширения
 */
CSInterface.prototype.closeExtension = function() {
    if (typeof window.__adobe_cep__ !== "undefined") {
        window.__adobe_cep__.closeExtension();
    }
};

/**
 * Запрос открытия URL
 */
CSInterface.prototype.openURLInDefaultBrowser = function(url) {
    if (typeof window.__adobe_cep__ !== "undefined") {
        window.__adobe_cep__.openURLInDefaultBrowser(url);
    } else {
        window.open(url);
    }
};

/**
 * Получение расширения по ID
 */
CSInterface.prototype.getExtensions = function(extensionIds) {
    if (typeof window.__adobe_cep__ !== "undefined") {
        return JSON.parse(window.__adobe_cep__.getExtensions(extensionIds));
    }
    return [];
};

/**
 * Получение ID сетевых настроек
 */
CSInterface.prototype.getNetworkPreferences = function() {
    if (typeof window.__adobe_cep__ !== "undefined") {
        return JSON.parse(window.__adobe_cep__.getNetworkPreferences());
    }
    return null;
};

/**
 * Регистрация обработчика события
 */
CSInterface.prototype.addEventListener = function(type, listener, obj) {
    if (typeof window.__adobe_cep__ !== "undefined") {
        window.__adobe_cep__.addEventListener(type, listener, obj);
    }
};

/**
 * Удаление обработчика события
 */
CSInterface.prototype.removeEventListener = function(type, listener, obj) {
    if (typeof window.__adobe_cep__ !== "undefined") {
        window.__adobe_cep__.removeEventListener(type, listener, obj);
    }
};

/**
 * Отправка события
 */
CSInterface.prototype.dispatchEvent = function(event) {
    if (typeof window.__adobe_cep__ !== "undefined") {
        window.__adobe_cep__.dispatchEvent(event);
    }
};

/**
 * Запрос контекстного меню
 */
CSInterface.prototype.setContextMenu = function(menu, callback) {
    if (typeof window.__adobe_cep__ !== "undefined") {
        window.__adobe_cep__.invokeAsync("setContextMenu", menu, callback);
    }
};

/**
 * Установка размера окна
 */
CSInterface.prototype.resizeContent = function(width, height) {
    if (typeof window.__adobe_cep__ !== "undefined") {
        window.__adobe_cep__.resizeContent(width, height);
    }
};

/**
 * CSEvent
 */
function CSEvent(type, scope, appId, extensionId) {
    this.type = type;
    this.scope = scope;
    this.appId = appId;
    this.extensionId = extensionId;
    this.data = "";
}
