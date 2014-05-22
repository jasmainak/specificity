clear all;

default_dir = '../';
target_dir = input(sprintf('Please enter target directory (Default %s): ', ...
                   default_dir), 's');

% Use default if not specified by user
if isempty(target_dir)
    target_dir = default_dir;
end

url = 'https://filebox.ece.vt.edu/~mainak/data/';

command = sprintf(['wget -r --no-parent --cut-dirs=1 -nv -nH -nc ' ...
                   '--reject="index.html*" -P %s ' ...
                   '--no-check-certificate %s'], target_dir, url);
system(command);
