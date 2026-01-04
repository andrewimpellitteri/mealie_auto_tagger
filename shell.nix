{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.uv
  ];

  shellHook = ''
    echo "Mealie Auto-Tagger Environment"
    echo "Recommended: uv run mealie-auto-tagger.py"
  '';
}
