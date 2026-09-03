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
; The same icon the application wears, taken from the payload rather than kept as a
; second copy that can drift from it. Without this, Setup.exe carried Inno's generic
; icon -- so the first thing anyone sees of this product looked like a stock installer
; belonging to nothing in particular.
SetupIconFile={#PayloadDir}\app.ico
; The application refuses a polite close on purpose: its window hides to the tray
; instead of exiting, because the service behind it holds loaded models for other
; programs. Restart Manager asks politely, gets refused, and Setup then aborts and
; rolls back -- so installing an update while the application was running did nothing
; at all, and under /VERYSILENT it did nothing silently.
;
; force closes it rather than asking. The backend is a separate process and survives,
; which is the wanted outcome twice over: models stay loaded for whatever else is
; using them, and the newly installed launcher adopts the running service on its next
; start rather than waiting for a port it cannot have.
CloseApplications=force
; Setup does not relaunch it. A launcher that reappears by itself after an install the
; user ran from Explorer is a surprise; the shortcut is right there.
RestartApplications=no
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

[UninstallDelete]
; The uninstaller removes what it recorded at install time and nothing else, and the
; backend writes __pycache__ into its own program directory as it runs. Without this
; the directory survives an uninstall holding a scattering of .pyc files -- observed,
; not hypothetical -- and an application that looks half-removed is worse than one
; that is still there.
Type: filesandordirs; Name: "{app}"

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

// "Label" is reserved in Pascal -- it names a goto target -- hence What.
procedure RemoveIfExists(Target: String; What: String);
begin
  if DirExists(Target) then
  begin
    Log('removing directory (' + What + '): ' + Target);
    DelTree(Target, True, True, True);
  end
  else if FileExists(Target) then
  begin
    Log('removing file (' + What + '): ' + Target);
    DeleteFile(Target);
  end;
end;

function ModelCacheDir(): String;
begin
  // {%HF_HOME} first because the launcher writes HF_HOME into the backend's .env and
  // a user may have pointed it elsewhere; the default is what Hugging Face uses.
  //
  // {%USERPROFILE} and not {userprofile}: the latter is not an Inno Setup constant,
  // and ExpandConstant raises on an unknown one. That raise happened here, inside
  // CurUninstallStepChanged, where an exception is fatal -- see below.
  Result := ExpandConstant('{%HF_HOME}');
  if Result = '' then
    Result := ExpandConstant('{%USERPROFILE}\.cache\huggingface');
end;

procedure CleanUpBeyondTheProgramDirectory();
var
  ModelCache: String;
begin
  // An autostart entry pointing at a removed executable fails silently at every
  // login, forever. It is written by the application at runtime rather than by this
  // installer, so it has to be cleared here explicitly.
  if RegValueExists(HKEY_CURRENT_USER, RunKey, RunValue) then
  begin
    Log('clearing autostart entry');
    RegDeleteValue(HKEY_CURRENT_USER, RunKey, RunValue);
  end;

  // Provisioning output that lives outside the program directory, so uninstalling
  // does not touch it unless we do it here: the virtual environment, the launcher's
  // state and logs, and the staging directory the llama installer uses. The llama
  // runtime itself now lands under the program directory and goes with it.
  RemoveIfExists(DataDir(), 'launcher data, virtual environment and logs');
  RemoveIfExists(ExpandConstant('{localappdata}\Microsoft\WindowsApps\llama.exe'), 'llama binary');
  RemoveIfExists(ExpandConstant('{localappdata}\llama-app'), 'llama staging');

  // The model cache is asked about rather than assumed. It is commonly tens of
  // gigabytes, it is shared with any other Hugging Face tooling on the machine, and
  // re-downloading it is expensive.
  ModelCache := ModelCacheDir();
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

procedure CurUninstallStepChanged(CurStep: TUninstallStep);
begin
  if CurStep <> usUninstall then
    Exit;

  // Guarded, because an exception raised here is fatal to the uninstall: the program
  // directory, the shortcuts and the registry entry are all removed *after* this
  // returns, so anything that throws leaves an application that is visibly installed
  // and no longer works.
  //
  // That is not a precaution. ExpandConstant on a constant that does not exist threw
  // exactly here, and the uninstall stopped having deleted four gigabytes of the
  // environment while leaving every sign of the application in place.
  //
  // Everything this procedure does is extra cleanup. None of it is worth abandoning
  // the uninstall for, so a failure is logged and the uninstall carries on.
  try
    CleanUpBeyondTheProgramDirectory();
  except
    Log('cleanup beyond the program directory failed: ' + GetExceptionMessage);
    Log('continuing with the uninstall regardless');
  end;
end;

const
  RequiredGB = 10;

function InitializeSetup(): Boolean;
var
  FreeBytes, TotalBytes: Int64;
  FreeGB: Integer;
begin
  Result := True;
  // Setup itself is small; what needs room is the first launch. Saying so before
  // the install is more useful than failing partway through a download.
  //
  // GetSpaceOnDisk64 rather than GetSpaceOnDisk: the latter reports into Cardinal,
  // and passing Int64 to it is a type mismatch the compiler rejects.
  if GetSpaceOnDisk64(ExpandConstant('{localappdata}'), FreeBytes, TotalBytes) then
  begin
    FreeGB := FreeBytes div 1073741824;
    if FreeGB < RequiredGB then
      Result := SuppressibleMsgBox(
        'Only ' + IntToStr(FreeGB) + ' GB is free on this drive.' + #13#10#13#10 +
        'First launch downloads the GPU runtime and needs about ' +
        IntToStr(RequiredGB) + ' GB. Continue anyway?',
        mbConfirmation, MB_YESNO, IDYES) = IDYES;
  end;
end;
