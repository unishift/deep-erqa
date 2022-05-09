"Turn on syntax highlight"
syntax on
"
"Tab settings"
set tabstop=4 
set softtabstop=4 
set expandtab
set autoindent
set shiftwidth=4

"# strings"
set number

"Mouse setting"
set mouse=a

"Folding"
set foldmethod=indent
set foldnestmax=10
set nofoldenable
set foldlevel=2

"Russian keymap"
set keymap=russian-jcukenwin
set iminsert=0
set imsearch=0
highlight lCursor guifg=NONE guibg=Cyan

"MASM settings"
function Masm()
  e ++enc=cp866
  set syntax=masm
  set softtabstop=4
  set shiftwidth=4
endfunction

autocmd BufRead,BufNewFile *.asm call Masm()

"Install vim-plug"
if empty(glob('~/.vim/autoload/plug.vim'))
  silent !curl -fLo ~/.vim/autoload/plug.vim --create-dirs
    \ https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
  autocmd VimEnter * PlugInstall --sync | source $MYVIMRC
endif

"Plugins"
call plug#begin()

"File tree"
Plug 'scrooloose/nerdtree'

"Commenting"
Plug 'tomtom/tcomment_vim'

"AutoComplete"
Plug 'Valloric/YouCompleteMe'

"Molokai colorscheme
Plug 'tomasr/molokai'
Plug 'crusoexia/vim-monokai'

"Syntax highlight
Plug 'sheerun/vim-polyglot'

"Git plugins
Plug 'airblade/vim-gitgutter'

"Oblivion colorscheme
Plug 'veloce/vim-aldmeris'

call plug#end()

"YCM configs"
let g:ycm_confirm_extra_conf = 0
let g:ycm_add_preview_to_completeopt = 0
let g:ycm_autoclose_preview_window_after_insertion = 1

"Colorscheme"
if &term =~ '256color'
        " Disable Background Color Erase (BCE) so that color schemes
        " work properly when Vim is used inside tmux and GNU screen.
        set t_ut=
endif

colorscheme monokai
" hi Normal ctermbg=NONE

let g:powerline_pycmd="py3"
