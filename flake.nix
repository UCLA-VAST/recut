{
  description = "recut-pipeline";
  inputs = { 
    unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
    mcp3d.url = "/home/kdmarrett/mcp3d";
    #mcp3d.url = "git+ssh://git@muyezhu/mcp3d";
  };

  outputs = inputs: {
    defaultPackage.x86_64-linux = import ./recut.nix { nixpkgs = inputs.unstable; mcp3d = inputs.mcp3d; };
  };
}
