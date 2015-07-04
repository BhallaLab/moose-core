# AUTHOR    : AVIRAL GOEL
# EMAIL-ID  : aviralg@ncbs.res.in

echo "Installing Dependencies for Ubuntu"
sudo -E apt-get install $(grep -vE "^\s*#" dependencies  | tr "\n" " ")
