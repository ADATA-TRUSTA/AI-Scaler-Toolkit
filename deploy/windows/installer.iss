; Windows installer.
;
; Per-user throughout: no elevation, installs under LocalAppData, and every shortcut
; and registry write is in the user's own hive. That is what makes the whole
; download-and-run flow work without an administrator, and it is only possible because
; we never run anyone else's installer (optional components are shown, not executed).
;
; The payload is built by deploy/package/make_payload.py, which has already picked the
; correct frontend package and rewritten its config.json. This script does not know
; about brands.

#ifndef PayloadDir
  #error Pass /DPayloadDir=<path to the built payload>
#endif

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#define AppName "AI Scaler Toolkit"
#define AppId "AIScalerToolkit"
#define Publisher "ADATA"
#define LauncherExe "AIScalerToolkit.exe"

[Setup]
AppId={{8B1C5E42-3D7A-4C19-9F2B-6A5E4D3C1B70}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#Publisher}
DefaultDirName={localappdata}\Programs\{#AppId}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Per-user install: no UAC prompt, and nothing lands outside the user's profile.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; The repository's own licence, not a copy kept beside the installer -- one file to
; keep current instead of two.
LicenseFile=..\..\LICENSE
OutputDir=Output
OutputBaseFilename=AIScalerToolkit_{#AppVersion}_Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#LauncherExe}
; The installer carries only the base. The GPU runtime is fetched on first launch, so
; this stays a small download instead of a 6 GB one.
AppComments=GPU runtime is downloaded on first launch, not bundled.

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#LauncherExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#LauncherExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#LauncherExe}"; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent

[Code]
const
  RunKey = 'Software\Microsoft\Windows\CurrentVersion\Run';
  RunValue = 'AIScalerToolkit';

function DataDir(): String;
begin
  Result := ExpandConstant('{localappdata}\AIScalerToolkit');
end;

procedure RemoveIfExists(Path: String; Label: String);
begin
  if DirExists(Path) then
  begin
    Log('removing directory (' + Label + '): ' + Path);
    DelTree(Path, True, True, True);
  end
  else if FileExists(Path) then
  begin
    Log('removing file (' + Label + '): ' + Path);
    DeleteFile(Path);
  end;
end;

procedure CurUninstallStepChanged(CurStep: TUninstallStep);
var
  ModelCache: String;
begin
  if CurStep <> usUninstall then
    Exit;

  { An autostart entry pointing at a removed executable fails silently at every
    login, forever. It is written by the application at runtime rather than by this
    installer, so it has to be cleared here explicitly. }
  if RegValueExists(HKEY_CURRENT_USER, RunKey, RunValue) then
  begin
    Log('clearing autostart entry');
    RegDeleteValue(HKEY_CURRENT_USER, RunKey, RunValue);
  end;

  { Provisioning output lives outside {app} on purpose, so uninstalling the program
    does not touch it unless we do it here: the virtual environment, the llama
    binary, the conversion tooling checkout, the installer's staging directory, and
    the launcher's own state and logs. }
  RemoveIfExists(DataDir(), 'launcher data, virtual environment and logs');
  RemoveIfExists(ExpandConstant('{localappdata}\Microsoft\WindowsApps\llama.exe'), 'llama binary');
  RemoveIfExists(ExpandConstant('{localappdata}\llama-app'), 'llama staging');

  { The model cache is asked about rather than assumed. It is commonly tens of
    gigabytes, it is shared with any other Hugging Face tooling on the machine, and
    re-downloading it is expensive. }
  ModelCache := ExpandConstant('{userprofile}\.cache\huggingface');
  if DirExists(ModelCache) then
  begin
    if SuppressibleMsgBox(
         'Also remove the downloaded models?' + #13#10#13#10 +
         ModelCache + #13#10#13#10 +
         'These can be several tens of gigabytes. Other tools on this machine may ' +
         'use the same cache, and keeping them means a reinstall does not download ' +
         'them again.',
         mbConfirmation, MB_YESNO, IDNO) = IDYES then
      RemoveIfExists(ModelCache, 'model cache')
    else
      Log('keeping the model cache at ' + ModelCache);
  end;
end;

function InitializeSetup(): Boolean;
var
  FreeMB, TotalMB: Int64;
begin
  Result := True;
  { Setup itself is small; what needs room is the first launch. Saying so before the
    install is more useful than failing partway through a multi-gigabyte download. }
  if GetSpaceOnDisk(ExpandConstant('{localappdata}'), True, FreeMB, TotalMB) then
  begin
    if FreeMB < 10240 then
      Result := SuppressibleMsgBox(
        'Only ' + IntToStr(FreeMB div 1024) + ' GB is free on this drive.' + #13#10#13#10 +
        'First launch downloads the GPU runtime and needs about 10 GB. Continue anyway?',
        mbConfirmation, MB_YESNO, IDYES) = IDYES;
  end;
end;
