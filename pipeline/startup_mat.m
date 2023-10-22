%Startup stuff
% Run script in code_matlab rep inside CIFAR project directroy
global CIFAR_version;
CIFAR_version.major = 1;
CIFAR_version.minor = 0;

rootdir = getenv('USERPROFILE');
%fprintf('[CIFAR startup] Initialising CIFAR version %d.%d\n', PsyMEG_version.major, PsyMEG_version.minor);

% Home dir

global home_dir CIFAR_root % parent folder of CIFAR directory
home_dir = fullfile('~');
CIFAR_root = fullfile(home_dir, 'projects', 'cifar') ; 
addpath(genpath(CIFAR_root));
rmpath(fullfile(CIFAR_root, 'scripts','deprecated')); % remove deprecated code to avoid confusion
fprintf('[CIFAR startup] Added path %s and appropriate subpaths\n',CIFAR_root);

% Initialize mvgc library

global mvgc_root;
mvgc_root = fullfile(home_dir,'toolboxes','MVGC2'); %or add MVGC_deprecated
assert(exist(mvgc_root,'dir') == 7,'bad MVGC path: ''%s'' does not exist or is not a directory',mvgc_root);
cd(mvgc_root);
startup;
cd(CIFAR_root);

% Add other useful toolboxes

global toolbox_dir ESN_dir noisetool eeglab_root LPZ_dir

cd(home_dir)
toolbox_dir = fullfile(home_dir,'toolboxes');
noisetool = fullfile(toolbox_dir,'NoiseTools');
eeglab_root = fullfile(toolbox_dir,'eeglab2019_1');
LPZ_dir = fullfile(toolbox_dir, 'fLZc');

addpath(genpath(fullfile(LPZ_dir)));
addpath(genpath(fullfile(ESN_dir)));
addpath(genpath(fullfile(noisetool)));
rmpath(fullfile(noisetool, 'COMPAT'));
addpath(genpath(fullfile(eeglab_root)));

cd(CIFAR_root)

% Get screen size

global screenxy

s = get(0,'ScreenSize');
screenxy = s([3 4]);

% Make all plot fonts bigger!

set(0,'DefaultAxesFontSize',20);
set(0,'DefaultTextFontSize',18);

% Make plot colours sane (like the old MATLAB)!

set(groot,'defaultAxesColorOrder',[0 0 1; 0 0.5 0; 1 0 0; 0 0.75 0.75; 0.75 0 0.75; 0.75 0.75 0; 0.25 0.25 0.25]);

% Dock figures
set(0,'DefaultFigureWindowStyle','docked')

fprintf('[CIFAR startup] Initialised (you may re-run `startup'' at any time)\n');
