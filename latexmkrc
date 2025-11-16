$aux_dir = 'tex/build';
$out_dir = 'docs/models';

$pdf_mode = 1;           # generate PDF
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# Ensure output dirs exist
sub ensure_dirs {
  my ($d) = @_;
  if (! -d $d) { system('mkdir','-p',$d); }
}
ensure_dirs($aux_dir);
ensure_dirs($out_dir);

# Quiet-ish by default
$silent = 0;

