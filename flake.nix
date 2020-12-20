{
  description = "recut-pipeline";
  inputs = { 
    unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
    mcp3d.url = "/home/kdmarrett/mcp3d";
    mcp3d.flake = false;
  };

  outputs = inputs: {
    defaultPackage.x86_64-linux = import ./recut.nix { nixpkgs = inputs.unstable; };
  };
}
