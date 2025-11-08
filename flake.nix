{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      devEnv = builtins.getEnv "NIX_DEV_ENV";

      torch =
        if devEnv == "desktop" then pkgs.python312Packages.torchWithCuda else pkgs.python312Packages.torch;
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python312
          python312Packages.matplotlib
          torch
          cudaPackages.cudatoolkit
        ];
      };
    };
}
