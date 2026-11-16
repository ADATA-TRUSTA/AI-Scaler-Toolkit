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
; The release build passes the full X.Y.Z.B (0.1.0.3), so Setup.exe's file properties
; name the exact build, not just the public version several builds can share.
VersionInfoVersion={#AppVersion}
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
; Product, version, architecture -- named like the AppImage, with Windows' word for it.
OutputBaseFilename=AIScalerToolkit-{#AppVersion}-x64-Setup
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
; force closes it rather than asking. Closed this way the launcher shuts down as on
; Quit, stopping a backend it started; one it had adopted from an earlier launcher is
; not its to stop and keeps running. The next launcher restarts that one if an older
; version started it (service_version in the launcher's state), so an update always
; ends with the new backend running and never the old one.
CloseApplications=force
; Setup does not relaunch it. A launcher that reappears by itself after an install the
; user ran from Explorer is a surprise; the shortcut is right there.
RestartApplications=no
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#LauncherExe}
; unins000.exe is written by Inno at compile time, from inside this script, so no
; signing step outside it can reach it; left alone it is the one unsigned program we
; install, and removing the application meets an "unknown publisher" prompt. Signing
; it needs a sign tool, which only the release build has: it passes
; /DSignToolName=<name> together with /S<name>=<command>. Without them the build is
; unsigned throughout, as a developer's should be.
#ifdef SignToolName
SignTool={#SignToolName}
SignedUninstaller=yes
#endif
; The installer carries only the base. The GPU runtime is fetched on first launch, so
; this stays a small download instead of a 6 GB one.
AppComments=GPU runtime is downloaded on first launch, not bundled.

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[InstallDelete]
; Installing adds and overwrites; it never removes something an older version put
; there. The console used to ship in the payload and no longer does, so without this
; an upgrade leaves it behind -- 67 files with no interpreter to run them, in the
; directory a user browses when looking for the application.
Type: filesandordirs; Name: "{app}\console"

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
; Through Explorer, not directly, so the application is not a child of Setup.
;
; Setup runs with the redirection-trust mitigation enabled -- reasonable for an
; installer, which must not be tricked into following a junction someone planted. A
; child process inherits it, and an application that inherits it cannot traverse a
; junction either. uv keeps its managed Python behind one: the interpreter lives at
; cpython-3.12.13-... and everything refers to it through the minor-version alias
; cpython-3.12-..., which on Windows is a junction.
;
; So ticking "Start AI Scaler Toolkit" on the last page produced a first run in which
; `uv sync` died instantly with ERROR_UNTRUSTED_MOUNT_POINT, no Python packages were
; installed, and setup reported success. Launching through Explorer gives the same
; process context as the desktop shortcut, which is what the user gets every other
; time they open it.
Filename: "{win}\explorer.exe"; Parameters: """{app}\{#LauncherExe}"""; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent

[Code]
const
  RunKey = 'Software\Microsoft\Windows\CurrentVersion\Run';
  RunValue = 'AIScalerToolkit';

function DataDir(): String;
begin
  Result := ExpandConstant('{localappdata}\AIScalerToolkit');
end;

// Empty a directory that DelTree could not, and remove it.
//
// The backend's environment holds files deeper than Windows' 260 characters: torch
// ships licence texts at ...\torch-*.dist-info\licenses\third_party\..., 254 characters
// plus the account's name under the default location -- measured at 258 for a name of
// four letters, so past the limit for one of six. uv writes them without trouble, and
// DelTree cannot delete them -- nor can it be handed the \\?\ form of the path, which
// it refuses outright. So an uninstall left the environment's deepest files in place
// with its interpreter gone, and the next install stopped at uv's "not a valid Python
// environment": uninstalling and installing again, the first thing anyone tries, was
// the one sequence that could not work.
//
// robocopy has no such limit, and mirroring an empty folder over a directory empties
// it. It leaves a junction's target alone, removing only the junction -- checked, since
// people do link a models folder to another drive. It only ever sees what DelTree left.
procedure EmptyOut(Dir: String);
var
  Empty: String;
  Code: Integer;
begin
  Empty := ExpandConstant('{tmp}\empty');
  if not ForceDirectories(Empty) then
  begin
    Log('could not create ' + Empty);
    Exit;
  end;
  // Below 8 is success for robocopy; 8 and above means something could not be removed.
  if not Exec(ExpandConstant('{sys}\robocopy.exe'),
              '"' + Empty + '" "' + Dir + '" /MIR /R:0 /W:0 /NFL /NDL /NJH /NJS /NP',
              '', SW_HIDE, ewWaitUntilTerminated, Code) then
    Log('robocopy could not be started: ' + SysErrorMessage(Code))
  else
    Log('robocopy exited ' + IntToStr(Code));
  RemoveDir(Dir);
end;

// "Label" is reserved in Pascal -- it names a goto target -- hence What.
procedure RemoveIfExists(Target: String; What: String);
begin
  Target := RemoveBackslashUnlessRoot(Target);
  if DirExists(Target) then
  begin
    Log('removing directory (' + What + '): ' + Target);
    DelTree(Target, True, True, True);
    // Never a drive root, whatever a path constant came to.
    if DirExists(Target) and (Length(Target) > 3) then
    begin
      Log('paths too long for DelTree are left in ' + Target + '; emptying it');
      EmptyOut(Target);
    end;
    if DirExists(Target) then
      Log('could not remove all of ' + Target);
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

// Every entry of Dir except the one named Keep, which stays with Dir around it.
// Listed first and removed after, rather than removed while the listing is still open.
procedure RemoveAllBut(Dir: String; Keep: String; What: String);
var
  FindRec: TFindRec;
  Names: TStringList;
  I: Integer;
begin
  Names := TStringList.Create;
  try
    if FindFirst(Dir + '\*', FindRec) then
    try
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') and
           (CompareText(FindRec.Name, Keep) <> 0) then
          Names.Add(FindRec.Name);
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
    for I := 0 to Names.Count - 1 do
      RemoveIfExists(Dir + '\' + Names[I], What);
  finally
    Names.Free;
  end;
end;

// ComfyUI keeps the user's models and everything it generates under the program
// directory ({app}\components\comfyui\ComfyUI), which [UninstallDelete] clears
// wholesale. Those are tens to hundreds of gigabytes the user put there, so they are
// asked about like the model cache, and the ones to keep are moved out of {app}
// beside it (same volume, so a rename and not a copy) before it is removed.
procedure KeepOrRemoveComfyData();
var
  ComfyDir: String;
  KeepDir: String;
  Names: array[0..1] of String;
  I: Integer;
  Src: String;
  Dest: String;
begin
  ComfyDir := ExpandConstant('{app}') + '\components\comfyui\ComfyUI';
  KeepDir := ExtractFilePath(RemoveBackslash(ExpandConstant('{app}'))) + 'AIScalerToolkit-kept';
  Names[0] := 'models';
  Names[1] := 'output';
  for I := 0 to 1 do
  begin
    Src := ComfyDir + '' + Names[I];
    if not DirExists(Src) then
      Continue;
    if SuppressibleMsgBox(
         'Also remove ComfyUI''s ' + Names[I] + '?' + #13#10#13#10 +
         Src + #13#10#13#10 +
         'These can be tens of gigabytes. Choose No to keep them: they are moved to ' +
         KeepDir + '.',
         mbConfirmation, MB_YESNO, IDNO) = IDYES then
    begin
      Log('removing ComfyUI ' + Names[I] + ' with the program directory');
      Continue;
    end;
    Dest := KeepDir + '\ComfyUI-' + Names[I];
    ForceDirectories(KeepDir);
    if DirExists(Dest) then
      Dest := Dest + '-' + GetDateTimeString('yyyymmdd-hhnnss', '-', ':');
    if RenameFile(Src, Dest) then
      Log('kept ComfyUI ' + Names[I] + ' at ' + Dest)
    else
    begin
      Log('could not move ' + Src + ' to ' + Dest);
      SuppressibleMsgBox(
        'Could not move ' + Src + ' out of the program directory, so it will be ' +
        'deleted with it.' + #13#10#13#10 +
        'Copy it somewhere else now if you want to keep it, then press OK.',
        mbError, MB_OK, IDOK);
    end;
  end;
end;

procedure CleanUpBeyondTheProgramDirectory();
var
  ModelCache: String;
  StrataData: String;
  KeepStrataData: Boolean;
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
  // does not touch it unless we do it here: the launcher's state and logs, and the
  // staging directory the llama installer uses. The llama runtime, the backend's
  // virtual environment, ComfyUI and Strata's engine all land under the program
  // directory now and go with it ([UninstallDelete] clears {app} wholesale). What
  // Strata derives from a model does not -- STRATA_DATA_DIR keeps Strata-data here,
  // because the question below offers to keep it. An install made before they moved
  // still holds the environment here, which is why it is still named below.
  // Strata's setup keeps what it derives from each model (the expert pack, the draft
  // layer) in Strata-data beside its install: gigabytes, and minutes of setup to make
  // again. Asked about like the model cache, so the rest of the data directory goes
  // either way.
  KeepOrRemoveComfyData();

  StrataData := DataDir() + '\Strata-data';
  KeepStrataData := False;
  if DirExists(StrataData) then
    KeepStrataData := SuppressibleMsgBox(
         'Also remove the data Strata prepared from your models?' + #13#10#13#10 +
         StrataData + #13#10#13#10 +
         'This can be several gigabytes. Strata''s setup made it from the downloaded ' +
         'models, and makes it again the next time a model is loaded with Strata.',
         mbConfirmation, MB_YESNO, IDNO) <> IDYES;
  if KeepStrataData then
  begin
    Log('keeping Strata''s data at ' + StrataData);
    RemoveAllBut(DataDir(), 'Strata-data', 'launcher data, logs and any environment an older install left here');
  end
  else
    RemoveIfExists(DataDir(), 'launcher data, logs and any environment an older install left here');
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

// What [UninstallDelete] could not take of the program directory: the paths past 260
// characters (see EmptyOut). After the uninstall proper, so everything it can remove
// in the ordinary way is already gone and this only has the remainder to deal with.
procedure RemoveWhatIsLeftOfTheProgramDirectory();
begin
  RemoveIfExists(ExpandConstant('{app}'), 'what was left of the program directory');
end;

procedure CurUninstallStepChanged(CurStep: TUninstallStep);
begin
  if CurStep = usPostUninstall then
  begin
    // Guarded like the cleanup below, and for the same reason: nothing here is worth
    // ending the uninstall on an error for.
    try
      RemoveWhatIsLeftOfTheProgramDirectory();
    except
      Log('removing what was left of the program directory failed: ' + GetExceptionMessage);
    end;
    Exit;
  end;

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
  UninstallKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{8B1C5E42-3D7A-4C19-9F2B-6A5E4D3C1B70}_is1';

var
  // The version already installed, or '' on a first install.
  PreviousVersion: String;

function InstalledVersion(): String;
begin
  // HKCU is where a per-user install registers itself. HKLM only in case an install
  // was ever made elevated, which would register there instead.
  if not RegQueryStringValue(HKCU, UninstallKey, 'DisplayVersion', Result) then
    if not RegQueryStringValue(HKLM, UninstallKey, 'DisplayVersion', Result) then
      Result := '';
end;

// The next dotted number from S, removed from S along with its dot. A part that is
// not a number counts as 0, so "0.4.13-rc1" compares as 0.4.13 and never throws.
function NextVersionPart(var S: String): Integer;
var
  Dot: Integer;
  Part: String;
begin
  Dot := Pos('.', S);
  if Dot = 0 then
  begin
    Part := S;
    S := '';
  end
  else
  begin
    Part := Copy(S, 1, Dot - 1);
    Delete(S, 1, Dot);
  end;
  Dot := Pos('-', Part);
  if Dot > 0 then
    Part := Copy(Part, 1, Dot - 1);
  Result := StrToIntDef(Trim(Part), 0);
end;

// Negative when A is older than B, 0 when the same, positive when newer.
function CompareVersions(A, B: String): Integer;
var
  PartA, PartB: Integer;
begin
  Result := 0;
  while (Result = 0) and ((A <> '') or (B <> '')) do
  begin
    PartA := NextVersionPart(A);
    PartB := NextVersionPart(B);
    Result := PartA - PartB;
  end;
end;

// Installing over an existing install replaces the program files and nothing else:
// the environment, the models and the settings live outside the program directory
// and stay. It used to happen without a word -- the same version reinstalled, an
// older one downgraded -- so each case is now said once, before anything changes.
//
// Silent installs (the updater, CI) ask nothing: the same or a newer version goes
// ahead, and an older one stops, because nothing unattended should downgrade.
function ConfirmOverExistingInstall(): Boolean;
var
  Order: Integer;
  Kept: String;
begin
  Result := True;
  PreviousVersion := InstalledVersion();
  if PreviousVersion = '' then
    Exit;

  Order := CompareVersions('{#AppVersion}', PreviousVersion);
  Log('installed version ' + PreviousVersion + ', this setup {#AppVersion}');
  if WizardSilent then
  begin
    Result := Order >= 0;
    if not Result then
      Log('not downgrading in a silent install; stopping');
    Exit;
  end;

  Kept := 'Your models, settings and the downloaded GPU runtime are kept.';
  if Order = 0 then
    Result := MsgBox(
      'AI Scaler Toolkit {#AppVersion} is already installed.' + #13#10#13#10 +
      'Reinstall its program files? ' + Kept,
      mbConfirmation, MB_YESNO) = IDYES
  else if Order > 0 then
    Result := MsgBox(
      'AI Scaler Toolkit will be updated from ' + PreviousVersion +
      ' to {#AppVersion}.' + #13#10#13#10 + Kept,
      mbInformation, MB_OKCANCEL) = IDOK
  else
    Result := MsgBox(
      'A newer version of AI Scaler Toolkit (' + PreviousVersion +
      ') is already installed.' + #13#10#13#10 +
      'Replace it with the older {#AppVersion}? Keep the newer one unless you ' +
      'know you need this version.',
      mbError, MB_YESNO or MB_DEFBUTTON2) = IDYES;
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  // The license was accepted when it was first installed.
  Result := (PageID = wpLicense) and (PreviousVersion <> '');
end;

function InitializeSetup(): Boolean;
var
  FreeBytes, TotalBytes: Int64;
  FreeGB: Integer;
begin
  Result := ConfirmOverExistingInstall();
  if not Result then
    Exit;
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
