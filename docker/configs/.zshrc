# The following lines were added by compinstall

zstyle ':completion:*' completer _expand _complete _ignored _approximate
zstyle ':completion:*' completions 1
zstyle ':completion:*' glob 1
# zstyle ':completion:*' matcher-list 'm:{[:lower:]}={[:upper:]}' 'r:|[._-]=** r:|=**'
zstyle ':completion:*' max-errors 2 numeric
zstyle ':completion:*' substitute 1
zstyle ':completion:*' menu select
zstyle :compinstall filename '/home/unishift/.zshrc'

autoload -Uz compinit
compinit
# End of lines added by compinstall
# Lines configured by zsh-newuser-install
HISTFILE=~/.histfile
HISTSIZE=1000
SAVEHIST=1000
setopt appendhistory extendedglob notify
unsetopt autocd beep nomatch
bindkey -v
# End of lines configured by zsh-newuser-install

setopt rmstarsilent

# Ctrl movements
bindkey "^[[1;5C" forward-word
bindkey "^[[1;5D" backward-word

source <(antibody init)
antibody bundle < ~/.zsh_plugins.txt

# source ~/.zsh_plugins.sh

alias dotfiles='git --git-dir=$HOME/.dotfiles/.git/ --work-tree=$HOME/.dotfiles/'
