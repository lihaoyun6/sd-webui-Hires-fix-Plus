import launch

if not launch.is_installed("tomesd"):
    launch.run_pip("install tomesd>=0.1.3", "requirements for Token Merging")
