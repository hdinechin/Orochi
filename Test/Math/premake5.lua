project "Math"
      kind "ConsoleApp"

      targetdir "../../dist/bin/%{cfg.buildcfg}"
      location "../../build/"

   if os.istarget("windows") then
      links{ "version" }
   end

      includedirs { "../../" }
      files { "../../Orochi/Orochi.h", "../../Orochi/Orochi.cpp" }
      files { "../../Orochi/OrochiUtils.h", "../../Orochi/OrochiUtils.cpp" }
      files { "../../contrib/**.h", "../../contrib/**.cpp" }
      files { "*.cpp" }
      files { "*.h" }
