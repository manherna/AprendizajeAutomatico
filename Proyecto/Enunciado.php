<!DOCTYPE html>

<html  dir="ltr" lang="es" xml:lang="es">
<head>
    <title>Campus Virtual Moodle 3.4: Iniciar sesión en el sitio</title>
    <link rel="shortcut icon" href="https://cv4.ucm.es/moodle/theme/image.php/boost/theme/1542011022/favicon" />
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="keywords" content="moodle, Campus Virtual Moodle 3.4: Iniciar sesión en el sitio" />
<link rel="stylesheet" type="text/css" href="https://cv4.ucm.es/moodle/theme/yui_combo.php?rollup/3.17.2/yui-moodlesimple-min.css" /><script id="firstthemesheet" type="text/css">/** Required in order to fix style inclusion problems in IE with YUI **/</script><link rel="stylesheet" type="text/css" href="https://cv4.ucm.es/moodle/theme/styles.php/boost/1542011022_1538482903/all" />
<script type="text/javascript">
//<![CDATA[
var M = {}; M.yui = {};
M.pageloadstarttime = new Date();
M.cfg = {"wwwroot":"https:\/\/cv4.ucm.es\/moodle","sesskey":"vQ6Otf85c9","themerev":"1542011022","slasharguments":1,"theme":"boost","iconsystemmodule":"core\/icon_system_fontawesome","jsrev":"1542011022","admin":"admin","svgicons":true,"usertimezone":"Europa\/Madrid","contextid":1};var yui1ConfigFn = function(me) {if(/-skin|reset|fonts|grids|base/.test(me.name)){me.type='css';me.path=me.path.replace(/\.js/,'.css');me.path=me.path.replace(/\/yui2-skin/,'/assets/skins/sam/yui2-skin')}};
var yui2ConfigFn = function(me) {var parts=me.name.replace(/^moodle-/,'').split('-'),component=parts.shift(),module=parts[0],min='-min';if(/-(skin|core)$/.test(me.name)){parts.pop();me.type='css';min=''}
if(module){var filename=parts.join('-');me.path=component+'/'+module+'/'+filename+min+'.'+me.type}else{me.path=component+'/'+component+'.'+me.type}};
YUI_config = {"debug":false,"base":"https:\/\/cv4.ucm.es\/moodle\/lib\/yuilib\/3.17.2\/","comboBase":"https:\/\/cv4.ucm.es\/moodle\/theme\/yui_combo.php?","combine":true,"filter":null,"insertBefore":"firstthemesheet","groups":{"yui2":{"base":"https:\/\/cv4.ucm.es\/moodle\/lib\/yuilib\/2in3\/2.9.0\/build\/","comboBase":"https:\/\/cv4.ucm.es\/moodle\/theme\/yui_combo.php?","combine":true,"ext":false,"root":"2in3\/2.9.0\/build\/","patterns":{"yui2-":{"group":"yui2","configFn":yui1ConfigFn}}},"moodle":{"name":"moodle","base":"https:\/\/cv4.ucm.es\/moodle\/theme\/yui_combo.php?m\/1542011022\/","combine":true,"comboBase":"https:\/\/cv4.ucm.es\/moodle\/theme\/yui_combo.php?","ext":false,"root":"m\/1542011022\/","patterns":{"moodle-":{"group":"moodle","configFn":yui2ConfigFn}},"filter":null,"modules":{"moodle-core-languninstallconfirm":{"requires":["base","node","moodle-core-notification-confirm","moodle-core-notification-alert"]},"moodle-core-formchangechecker":{"requires":["base","event-focus","moodle-core-event"]},"moodle-core-event":{"requires":["event-custom"]},"moodle-core-notification":{"requires":["moodle-core-notification-dialogue","moodle-core-notification-alert","moodle-core-notification-confirm","moodle-core-notification-exception","moodle-core-notification-ajaxexception"]},"moodle-core-notification-dialogue":{"requires":["base","node","panel","escape","event-key","dd-plugin","moodle-core-widget-focusafterclose","moodle-core-lockscroll"]},"moodle-core-notification-alert":{"requires":["moodle-core-notification-dialogue"]},"moodle-core-notification-confirm":{"requires":["moodle-core-notification-dialogue"]},"moodle-core-notification-exception":{"requires":["moodle-core-notification-dialogue"]},"moodle-core-notification-ajaxexception":{"requires":["moodle-core-notification-dialogue"]},"moodle-core-tooltip":{"requires":["base","node","io-base","moodle-core-notification-dialogue","json-parse","widget-position","widget-position-align","event-outside","cache-base"]},"moodle-core-popuphelp":{"requires":["moodle-core-tooltip"]},"moodle-core-blocks":{"requires":["base","node","io","dom","dd","dd-scroll","moodle-core-dragdrop","moodle-core-notification"]},"moodle-core-checknet":{"requires":["base-base","moodle-core-notification-alert","io-base"]},"moodle-core-handlebars":{"condition":{"trigger":"handlebars","when":"after"}},"moodle-core-maintenancemodetimer":{"requires":["base","node"]},"moodle-core-dragdrop":{"requires":["base","node","io","dom","dd","event-key","event-focus","moodle-core-notification"]},"moodle-core-dock":{"requires":["base","node","event-custom","event-mouseenter","event-resize","escape","moodle-core-dock-loader","moodle-core-event"]},"moodle-core-dock-loader":{"requires":["escape"]},"moodle-core-lockscroll":{"requires":["plugin","base-build"]},"moodle-core-chooserdialogue":{"requires":["base","panel","moodle-core-notification"]},"moodle-core-actionmenu":{"requires":["base","event","node-event-simulate"]},"moodle-core_availability-form":{"requires":["base","node","event","event-delegate","panel","moodle-core-notification-dialogue","json"]},"moodle-backup-backupselectall":{"requires":["node","event","node-event-simulate","anim"]},"moodle-backup-confirmcancel":{"requires":["node","node-event-simulate","moodle-core-notification-confirm"]},"moodle-course-formatchooser":{"requires":["base","node","node-event-simulate"]},"moodle-course-categoryexpander":{"requires":["node","event-key"]},"moodle-course-modchooser":{"requires":["moodle-core-chooserdialogue","moodle-course-coursebase"]},"moodle-course-management":{"requires":["base","node","io-base","moodle-core-notification-exception","json-parse","dd-constrain","dd-proxy","dd-drop","dd-delegate","node-event-delegate"]},"moodle-course-dragdrop":{"requires":["base","node","io","dom","dd","dd-scroll","moodle-core-dragdrop","moodle-core-notification","moodle-course-coursebase","moodle-course-util"]},"moodle-course-util":{"requires":["node"],"use":["moodle-course-util-base"],"submodules":{"moodle-course-util-base":{},"moodle-course-util-section":{"requires":["node","moodle-course-util-base"]},"moodle-course-util-cm":{"requires":["node","moodle-course-util-base"]}}},"moodle-form-shortforms":{"requires":["node","base","selector-css3","moodle-core-event"]},"moodle-form-dateselector":{"requires":["base","node","overlay","calendar"]},"moodle-form-showadvanced":{"requires":["node","base","selector-css3"]},"moodle-form-passwordunmask":{"requires":[]},"moodle-question-searchform":{"requires":["base","node"]},"moodle-question-qbankmanager":{"requires":["node","selector-css3"]},"moodle-question-preview":{"requires":["base","dom","event-delegate","event-key","core_question_engine"]},"moodle-question-chooser":{"requires":["moodle-core-chooserdialogue"]},"moodle-availability_completion-form":{"requires":["base","node","event","moodle-core_availability-form"]},"moodle-availability_date-form":{"requires":["base","node","event","io","moodle-core_availability-form"]},"moodle-availability_grade-form":{"requires":["base","node","event","moodle-core_availability-form"]},"moodle-availability_group-form":{"requires":["base","node","event","moodle-core_availability-form"]},"moodle-availability_grouping-form":{"requires":["base","node","event","moodle-core_availability-form"]},"moodle-availability_profile-form":{"requires":["base","node","event","moodle-core_availability-form"]},"moodle-qtype_ddimageortext-form":{"requires":["moodle-qtype_ddimageortext-dd","form_filepicker"]},"moodle-qtype_ddimageortext-dd":{"requires":["node","dd","dd-drop","dd-constrain"]},"moodle-qtype_ddmarker-form":{"requires":["moodle-qtype_ddmarker-dd","form_filepicker","graphics","escape"]},"moodle-qtype_ddmarker-dd":{"requires":["node","event-resize","dd","dd-drop","dd-constrain","graphics"]},"moodle-qtype_ddwtos-dd":{"requires":["node","dd","dd-drop","dd-constrain"]},"moodle-mod_assign-history":{"requires":["node","transition"]},"moodle-mod_bigbluebuttonbn-modform":{"requires":["base","node"]},"moodle-mod_bigbluebuttonbn-rooms":{"requires":["base","node","datasource-get","datasource-jsonschema","datasource-polling","moodle-core-notification"]},"moodle-mod_bigbluebuttonbn-recordings":{"requires":["base","node","datasource-get","datasource-jsonschema","datasource-polling","moodle-core-notification"]},"moodle-mod_bigbluebuttonbn-broker":{"requires":["base","node","datasource-get","datasource-jsonschema","datasource-polling","moodle-core-notification"]},"moodle-mod_bigbluebuttonbn-imports":{"requires":["base","node"]},"moodle-mod_forum-subscriptiontoggle":{"requires":["base-base","io-base"]},"moodle-mod_quiz-repaginate":{"requires":["base","event","node","io","moodle-core-notification-dialogue"]},"moodle-mod_quiz-quizbase":{"requires":["base","node"]},"moodle-mod_quiz-modform":{"requires":["base","node","event"]},"moodle-mod_quiz-toolboxes":{"requires":["base","node","event","event-key","io","moodle-mod_quiz-quizbase","moodle-mod_quiz-util-slot","moodle-core-notification-ajaxexception"]},"moodle-mod_quiz-randomquestion":{"requires":["base","event","node","io","moodle-core-notification-dialogue"]},"moodle-mod_quiz-dragdrop":{"requires":["base","node","io","dom","dd","dd-scroll","moodle-core-dragdrop","moodle-core-notification","moodle-mod_quiz-quizbase","moodle-mod_quiz-util-base","moodle-mod_quiz-util-page","moodle-mod_quiz-util-slot","moodle-course-util"]},"moodle-mod_quiz-quizquestionbank":{"requires":["base","event","node","io","io-form","yui-later","moodle-question-qbankmanager","moodle-core-notification-dialogue"]},"moodle-mod_quiz-autosave":{"requires":["base","node","event","event-valuechange","node-event-delegate","io-form"]},"moodle-mod_quiz-questionchooser":{"requires":["moodle-core-chooserdialogue","moodle-mod_quiz-util","querystring-parse"]},"moodle-mod_quiz-util":{"requires":["node","moodle-core-actionmenu"],"use":["moodle-mod_quiz-util-base"],"submodules":{"moodle-mod_quiz-util-base":{},"moodle-mod_quiz-util-slot":{"requires":["node","moodle-mod_quiz-util-base"]},"moodle-mod_quiz-util-page":{"requires":["node","moodle-mod_quiz-util-base"]}}},"moodle-message_airnotifier-toolboxes":{"requires":["base","node","io"]},"moodle-filter_glossary-autolinker":{"requires":["base","node","io-base","json-parse","event-delegate","overlay","moodle-core-event","moodle-core-notification-alert","moodle-core-notification-exception","moodle-core-notification-ajaxexception"]},"moodle-filter_mathjaxloader-loader":{"requires":["moodle-core-event"]},"moodle-editor_atto-rangy":{"requires":[]},"moodle-editor_atto-editor":{"requires":["node","transition","io","overlay","escape","event","event-simulate","event-custom","node-event-html5","node-event-simulate","yui-throttle","moodle-core-notification-dialogue","moodle-core-notification-confirm","moodle-editor_atto-rangy","handlebars","timers","querystring-stringify"]},"moodle-editor_atto-plugin":{"requires":["node","base","escape","event","event-outside","handlebars","event-custom","timers","moodle-editor_atto-menu"]},"moodle-editor_atto-menu":{"requires":["moodle-core-notification-dialogue","node","event","event-custom"]},"moodle-format_grid-gridkeys":{"requires":["event-nav-keys"]},"moodle-report_eventlist-eventfilter":{"requires":["base","event","node","node-event-delegate","datatable","autocomplete","autocomplete-filters"]},"moodle-report_loglive-fetchlogs":{"requires":["base","event","node","io","node-event-delegate"]},"moodle-gradereport_grader-gradereporttable":{"requires":["base","node","event","handlebars","overlay","event-hover"]},"moodle-gradereport_history-userselector":{"requires":["escape","event-delegate","event-key","handlebars","io-base","json-parse","moodle-core-notification-dialogue"]},"moodle-tool_capability-search":{"requires":["base","node"]},"moodle-tool_lp-dragdrop-reorder":{"requires":["moodle-core-dragdrop"]},"moodle-tool_monitor-dropdown":{"requires":["base","event","node"]},"moodle-assignfeedback_editpdf-editor":{"requires":["base","event","node","io","graphics","json","event-move","event-resize","transition","querystring-stringify-simple","moodle-core-notification-dialog","moodle-core-notification-alert","moodle-core-notification-exception","moodle-core-notification-ajaxexception"]},"moodle-atto_accessibilitychecker-button":{"requires":["color-base","moodle-editor_atto-plugin"]},"moodle-atto_accessibilityhelper-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_align-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_bold-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_charmap-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_clear-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_collapse-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_emoticon-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_equation-button":{"requires":["moodle-editor_atto-plugin","moodle-core-event","io","event-valuechange","tabview","array-extras"]},"moodle-atto_html-button":{"requires":["moodle-editor_atto-plugin","event-valuechange"]},"moodle-atto_image-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_indent-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_italic-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_link-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_managefiles-usedfiles":{"requires":["node","escape"]},"moodle-atto_managefiles-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_media-button":{"requires":["moodle-editor_atto-plugin","moodle-form-shortforms"]},"moodle-atto_noautolink-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_orderedlist-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_rtl-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_strike-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_subscript-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_superscript-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_table-button":{"requires":["moodle-editor_atto-plugin","moodle-editor_atto-menu","event","event-valuechange"]},"moodle-atto_title-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_underline-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_undo-button":{"requires":["moodle-editor_atto-plugin"]},"moodle-atto_unorderedlist-button":{"requires":["moodle-editor_atto-plugin"]}}},"gallery":{"name":"gallery","base":"https:\/\/cv4.ucm.es\/moodle\/lib\/yuilib\/gallery\/","combine":true,"comboBase":"https:\/\/cv4.ucm.es\/moodle\/theme\/yui_combo.php?","ext":false,"root":"gallery\/1542011022\/","patterns":{"gallery-":{"group":"gallery"}}}},"modules":{"core_filepicker":{"name":"core_filepicker","fullpath":"https:\/\/cv4.ucm.es\/moodle\/lib\/javascript.php\/1542011022\/repository\/filepicker.js","requires":["base","node","node-event-simulate","json","async-queue","io-base","io-upload-iframe","io-form","yui2-treeview","panel","cookie","datatable","datatable-sort","resize-plugin","dd-plugin","escape","moodle-core_filepicker","moodle-core-notification-dialogue"]},"core_comment":{"name":"core_comment","fullpath":"https:\/\/cv4.ucm.es\/moodle\/lib\/javascript.php\/1542011022\/comment\/comment.js","requires":["base","io-base","node","json","yui2-animation","overlay","escape"]},"mathjax":{"name":"mathjax","fullpath":"https:\/\/cdnjs.cloudflare.com\/ajax\/libs\/mathjax\/2.7.2\/MathJax.js?delayStartupUntil=configured"}}};
M.yui.loader = {modules: {}};

//]]>
</script>

<meta name="robots" content="noindex" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>

<body  id="page-login-index" class="format-site  path-login safari dir-ltr lang-es yui-skin-sam yui3-skin-sam cv4-ucm-es--moodle pagelayout-login course-1 context-1 notloggedin ">

<div id="page-wrapper">

    <div>
    <a class="sr-only sr-only-focusable" href="#maincontent">Saltar a contenido principal</a>
</div><script type="text/javascript" src="https://cv4.ucm.es/moodle/theme/yui_combo.php?rollup/3.17.2/yui-moodlesimple-min.js"></script><script type="text/javascript" src="https://cv4.ucm.es/moodle/lib/javascript.php/1542011022/lib/javascript-static.js"></script>
<script type="text/javascript">
//<![CDATA[
document.body.className += ' jsenabled';
//]]>
</script>



    <div id="page" class="container-fluid">
        <div id="page-content" class="row">
            <div id="region-main-box">
                <section id="region-main" class="col-xs-12">
                    <span class="notifications" id="user-notifications"></span>
                    <div role="main"><span id="maincontent"></span><div class="m-y-3 hidden-sm-down"></div>
<div class="row">
<div class="col-xl-6 push-xl-3 m-2-md col-sm-8 push-sm-2">
<div class="card">
    <div class="card-block">
        <div class="card-title text-xs-center">
                <h2><img src="https://cv4.ucm.es/moodle/pluginfile.php/1/core_admin/logo/0x200/1542011022/Marca%20UCM%20Secundaria%20Monocromo%20Negro.png" title="Campus Virtual Moodle 3.4" alt="Campus Virtual Moodle 3.4"/></h2>
            <hr>
        </div>


            <div class="loginerrors m-t-1">
                <a href="#" id="loginerrormessage" class="accesshide">Su sesión ha excedido el tiempo límite. Por favor, ingrese de nuevo.</a>
                <div class="alert alert-danger" role="alert">Su sesión ha excedido el tiempo límite. Por favor, ingrese de nuevo.</div>
            </div>

        <div class="row">
            <div class="col-md-4 push-md-1">
                <form class="m-t-1" action="https://cv4.ucm.es/moodle/login/index.php" method="post" id="login">
                    <input id="anchor" type="hidden" name="anchor" value="">
                    <script>document.getElementById('anchor').value = location.hash;</script>

                    <label for="username" class="sr-only">
                            Id. Usuario UCM
                    </label>
                    <input type="text" name="username" id="username"
                        class="form-control"
                        value="569363"
                        placeholder="Id. Usuario UCM">
                    <label for="password" class="sr-only">Contraseña</label>
                    <input type="password" name="password" id="password" value=""
                        class="form-control"
                        placeholder="Contraseña">

                        <div class="rememberpass m-t-1">
                            <input type="checkbox" name="rememberusername" id="rememberusername" value="1" checked="checked" />
                            <label for="rememberusername">Recordar nombre de usuario</label>
                        </div>

                    <button type="submit" class="btn btn-primary btn-block m-t-1" id="loginbtn">Acceder</button>
                </form>
            </div>

            <div class="col-md-4 push-md-3">
                <div class="forgetpass m-t-1">
                    <p><a href="https://cv4.ucm.es/moodle/login/forgot_password.php">¿Olvidó su nombre de usuario o contraseña?</a></p>
                </div>

                <div class="m-t-1">
                    Las 'Cookies' deben estar habilitadas en su navegador
                    <a class="btn btn-link p-a-0" role="button"
    data-container="body" data-toggle="popover"
    data-placement="right" data-content="&lt;div class=&quot;no-overflow&quot;&gt;&lt;p&gt;Este sitio utiliza dos &quot;cookies&quot;.&lt;/p&gt;

&lt;p&gt;La esencial es la de sesión, normalmente llamada &lt;b&gt;MoodleSession&lt;/b&gt;.
Debe permitir que su navegador la acepte para poder mantener el servicio
funcionando de una página a otra. Cuando sale de la plataforma o cierra su navegador la &#039;cookie&#039; se destruye (en su navegador y en el servidor).&lt;/p&gt;

&lt;p&gt;La otra &#039;cookie&#039;, normalmente llamada &lt;b&gt;MOODLEID&lt;/b&gt;, es para su comodidad. Se limita a recordar su nombre de usuario dentro del navegador. Esto significa que cuando
regrese al sitio se escribirá automáticamente su nombre en el campo nombre de usuario
(userid). Si desea mayor seguridad no utilice esta opción: sólo tendrá que escribir su nombre manualmente cada vez que quiera ingresar.&lt;/p&gt;
&lt;/div&gt; "
    data-html="true" tabindex="0" data-trigger="focus">
  <i class="icon fa fa-question-circle text-info fa-fw " aria-hidden="true" title="Ayuda con Las &#039;Cookies&#039; deben estar habilitadas en su navegador" aria-label="Ayuda con Las &#039;Cookies&#039; deben estar habilitadas en su navegador"></i>
</a>
                </div>
                <div class="m-t-2">
                    <p>Algunos cursos permiten el acceso de invitados</p>
                    <form action="https://cv4.ucm.es/moodle/login/index.php" method="post" id="guestlogin">
                        <input type="hidden" name="username" value="guest" />
                        <input type="hidden" name="password" value="guest" />
                        <button class="btn btn-secondary btn-block" type="submit">Iniciar sesión como invitado</button>
                    </form>
                </div>

                <h6 class="m-t-2">Identifíquese usando su cuenta en:</h6>
                <div class="potentialidplist" class="m-t-1">
                        <div class="potentialidp">
                            <a href="https://cv4.ucm.es/moodle/auth/saml2/login.php?wants=https%3A%2F%2Fcv4.ucm.es%2Fmoodle%2Fpluginfile.php%2F5760048%2Fmod_resource%2Fcontent%2F0%2Fproyecto.pdf&amp;idp=e78505e2f935fb6b1f051035c553ed4a&amp;passive=off" title="Acceso con cuenta UCM" class="btn btn-secondary btn-block">
                                    <img src="https://cv4.ucm.es/moodle/theme/image.php/boost/core/1542011022/i/user" alt="" width="24" height="24"/>
                                Acceso con cuenta UCM
                            </a>
                        </div>
                </div>
            </div>
        </div>
    </div>
</div>
</div>
</div></div>
                    
                </section>
            </div>
        </div>
    </div>
</div>
<footer id="page-footer" class="p-y-1 bg-inverse">
    <div class="container">
        <div id="course-footer"></div>


        <div class="logininfo">Usted no se ha identificado.</div>
        <div class="homelink"><a href="https://cv4.ucm.es/moodle/">Página Principal</a></div><div id="node_id" style="text-align:center;visibility: hidden;">NODO: 86</div>
        <a href="https://download.moodle.org/mobile?version=2017111303.05&amp;lang=es&amp;iosappid=633359593&amp;androidappid=com.moodle.moodlemobile">Descargar la app para dispositivos móviles</a>
        
<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-48858766-1', 'ucm.es');
  ga('send', 'pageview');

</script><script type="text/javascript">
//<![CDATA[
var require = {
    baseUrl : 'https://cv4.ucm.es/moodle/lib/requirejs.php/1542011022/',
    // We only support AMD modules with an explicit define() statement.
    enforceDefine: true,
    skipDataMain: true,
    waitSeconds : 0,

    paths: {
        jquery: 'https://cv4.ucm.es/moodle/lib/javascript.php/1542011022/lib/jquery/jquery-3.2.1.min',
        jqueryui: 'https://cv4.ucm.es/moodle/lib/javascript.php/1542011022/lib/jquery/ui-1.12.1/jquery-ui.min',
        jqueryprivate: 'https://cv4.ucm.es/moodle/lib/javascript.php/1542011022/lib/requirejs/jquery-private'
    },

    // Custom jquery config map.
    map: {
      // '*' means all modules will get 'jqueryprivate'
      // for their 'jquery' dependency.
      '*': { jquery: 'jqueryprivate' },
      // Stub module for 'process'. This is a workaround for a bug in MathJax (see MDL-60458).
      '*': { process: 'core/first' },

      // 'jquery-private' wants the real jQuery module
      // though. If this line was not here, there would
      // be an unresolvable cyclic dependency.
      jqueryprivate: { jquery: 'jquery' }
    }
};

//]]>
</script>
<script type="text/javascript" src="https://cv4.ucm.es/moodle/lib/javascript.php/1542011022/lib/requirejs/require.min.js"></script>
<script type="text/javascript">
//<![CDATA[
require(['core/first'], function() {
;
require(["media_videojs/loader"], function(loader) {
    loader.setUp(function(videojs) {
        videojs.options.flash.swf = "https://cv4.ucm.es/moodle/media/player/videojs/videojs/video-js.swf";
videojs.addLanguage("es",{
 "Play": "Reproducción",
 "Play Video": "Reproducción Vídeo",
 "Pause": "Pausa",
 "Current Time": "Tiempo reproducido",
 "Duration Time": "Duración total",
 "Remaining Time": "Tiempo restante",
 "Stream Type": "Tipo de secuencia",
 "LIVE": "DIRECTO",
 "Loaded": "Cargado",
 "Progress": "Progreso",
 "Fullscreen": "Pantalla completa",
 "Non-Fullscreen": "Pantalla no completa",
 "Mute": "Silenciar",
 "Unmute": "No silenciado",
 "Playback Rate": "Velocidad de reproducción",
 "Subtitles": "Subtítulos",
 "subtitles off": "Subtítulos desactivados",
 "Captions": "Subtítulos especiales",
 "captions off": "Subtítulos especiales desactivados",
 "Chapters": "Capítulos",
 "You aborted the media playback": "Ha interrumpido la reproducción del vídeo.",
 "A network error caused the media download to fail part-way.": "Un error de red ha interrumpido la descarga del vídeo.",
 "The media could not be loaded, either because the server or network failed or because the format is not supported.": "No se ha podido cargar el vídeo debido a un fallo de red o del servidor o porque el formato es incompatible.",
 "The media playback was aborted due to a corruption problem or because the media used features your browser did not support.": "La reproducción de vídeo se ha interrumpido por un problema de corrupción de datos o porque el vídeo precisa funciones que su navegador no ofrece.",
 "No compatible source was found for this media.": "No se ha encontrado ninguna fuente compatible con este vídeo."
});

    });
});;

require(['theme_boost/loader']);
;

        require(['jquery'], function($) {
            $('#loginerrormessage').focus();
        });
;
require(["core/notification"], function(amd) { amd.init(1, []); });;
require(["core/log"], function(amd) { amd.setConfig({"level":"warn"}); });
});
//]]>
</script>
<script type="text/javascript">
//<![CDATA[
M.str = {"moodle":{"lastmodified":"\u00daltima modificaci\u00f3n","name":"Nombre","error":"Error","info":"Informaci\u00f3n","yes":"S\u00ed","no":"No","ok":"OK","cancel":"Cancelar","confirm":"Confirmar","areyousure":"\u00bfEst\u00e0 seguro?","closebuttontitle":"Cerrar","unknownerror":"Error desconocido"},"repository":{"type":"Tipo","size":"Tama\u00f1o","invalidjson":"Cadena JSON no v\u00e1lida","nofilesattached":"No se han adjuntado archivos","filepicker":"Selector de archivos","logout":"Salir","nofilesavailable":"No hay archivos disponibles","norepositoriesavailable":"Lo sentimos, ninguno de sus repositorios actuales puede devolver archivos en el formato solicitado.","fileexistsdialogheader":"El archivo existe","fileexistsdialog_editor":"Un archivo con el mismo nombre ya se ha adjuntado al texto que est\u00e1 editando.","fileexistsdialog_filemanager":"Un archivo con ese nombre ya ha sido adjuntado","renameto":"Cambiar el nombre a \"{$a}\"","referencesexist":"Existen {$a} archivos de alias\/atajos que emplean este archivo como su or\u00edgen","select":"Seleccionar"},"admin":{"confirmdeletecomments":"Est\u00e1 a punto de eliminar comentarios, \u00bfest\u00e1 seguro?","confirmation":"Confirmaci\u00f3n"}};
//]]>
</script>
<script type="text/javascript">
//<![CDATA[
(function() {Y.use("moodle-filter_glossary-autolinker",function() {M.filter_glossary.init_filter_autolinking({"courseid":0});
});
Y.use("moodle-filter_mathjaxloader-loader",function() {M.filter_mathjaxloader.configure({"mathjaxconfig":"\nMathJax.Hub.Config({\n    config: [\"Accessible.js\", \"Safe.js\"],\n    errorSettings: { message: [\"!\"] },\n    skipStartupTypeset: true,\n    messageStyle: \"none\"\n});\n","lang":"es"});
});
M.util.help_popups.setup(Y);
 M.util.js_pending('random5c3b5f709e7d02'); Y.on('domready', function() { M.util.js_complete("init");  M.util.js_complete('random5c3b5f709e7d02'); });
})();
//]]>
</script>

    </div>
</footer>

</body>
</html>