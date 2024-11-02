@echo off
setlocal enabledelayedexpansion

REM Definiere die Variablen
set dataset_path=data\unity_fuwa_small_1
set dataset_path_from_sub=..\data\unity_fuwa_small_1
set casename=unity_fuwa_small_1
@REM set gt_folder=..\data\label
set root_path=..

REM Funktion zum Beenden bei Fehler
set exitOnError=if errorlevel 1 exit /b

REM 1. Get the language feature of the scene
echo "start preprocessing!!!!!!!!!!!!!!!"
echo python preprocess.py --dataset_path %dataset_path%
python preprocess.py --dataset_path %dataset_path%
%exitOnError%

REM 2. Train the autoencoder
cd autoencoder
echo "start autoencoding training!!!!!!!!!!!!!!!"
echo python train.py --dataset_path %dataset_path_from_sub% --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name %casename%
python train.py --dataset_path %dataset_path_from_sub% --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name %casename%
%exitOnError%

REM Ordner erstellen und language_features kopieren
set new_folder=%dataset_path_from_sub%/language_features_dim3
echo Erstelle Ordner: %new_folder%
mkdir "%new_folder%"
%exitOnError%

echo Kopiere language_features nach %new_folder%
xcopy /E /I "%dataset_path_from_sub%/language_features" "%new_folder%/language_features"
%exitOnError%

REM 3. Get the 3-dims language feature of the scene
echo python test.py --dataset_path %dataset_path_from_sub% --dataset_name %casename%
python test.py --dataset_path %dataset_path_from_sub% --dataset_name %casename%
%exitOnError%

REM Kopieren der Dateien mit _f.npy-Endung und Löschen des Ursprungsordners
set source_folder=%dataset_path_from_sub%/language_features_dim3/language_features
set target_folder=%dataset_path_from_sub%/language_features_dim3

echo Kopiere Dateien mit der Endung _f.npy von %source_folder% nach %target_folder%
for %%F in (%source_folder%\*_f.npy) do (
    copy "%%F" "%target_folder%"
    %exitOnError%
)

REM Löschen des Ursprungsordners, nachdem alle Dateien kopiert wurden
echo Lösche Ordner %source_folder%
rd /S /Q "%source_folder%"
%exitOnError%
cd ..

REM Schleife für verschiedene Level
for %%L in (1 2 3) do (
    echo python train.py -s %dataset_path% -m output/%casename% --start_checkpoint %dataset_path%/%casename%/chkpnt30000.pth --feature_level %%L
    %exitOnError%
    python train.py -s %dataset_path% -m output/%casename% --start_checkpoint %dataset_path%/%casename%/chkpnt30000.pth --feature_level %%L
    %exitOnError%
)

REM Rendern für verschiedene Level
for %%L in (1 2 3) do (
    echo python render.py -m output/%casename%_%%L
    python render.py -m output/%casename%_%%L
    %exitOnError%

    @REM echo python render.py -m output/%casename%_%%L --include_feature
    @REM python render.py -m output/%casename%_%%L --include_feature
    @REM %exitOnError%
)

@REM cd eval
@REM echo Evaluate that stuff
@REM echo python just_render.py --dataset_name %casename% --feat_dir %root_path%\output --ae_ckpt_dir %root_path%\autoencoder\ckpt --output_dir %root_path%\eval_result --mask_thresh 0.4 --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512
@REM python just_render.py --dataset_name %casename% --feat_dir %root_path%\output --ae_ckpt_dir %root_path%\autoencoder\ckpt --output_dir %root_path%\eval_result --mask_thresh 0.4 --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512