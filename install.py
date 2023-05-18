import launch

if not launch.is_installed("aitextgen"):
    launch.run_pip("install aitextgen>=0.1.3", "requirements for Token Merging")
