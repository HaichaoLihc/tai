

 var \_gaq = \_gaq || [];
 \_gaq.push(['\_setAccount', 'UA-17821189-2']);
 \_gaq.push(['\_setDomainName', 'wiki.ros.org']);
 \_gaq.push(['\_trackPageview']);

 (function() {
 var ga = document.createElement('script'); ga.type = 'text/javascript'; ga.async = true;
 ga.src = ('https:' == document.location.protocol ? 'https://ssl' : 'http://www') + '.google-analytics.com/ga.js';
 var s = document.getElementsByTagName('script')[0]; s.parentNode.insertBefore(ga, s);
 })();

ROS/Tutorials/rosbuild/NavigatingTheFilesystem - ROS Wiki

<!--
var search\_hint = "Search";
//-->

 window.dataLayer = window.dataLayer || [];
 function gtag(){dataLayer.push(arguments);}
 gtag('js', new Date());

 gtag('config', 'G-EVD5Z6G6NH');

<!--// Initialize search form
var f = document.getElementById('searchform');
if(f) f.getElementsByTagName('label')[0].style.display = 'none';
var e = document.getElementById('searchinput');
if(e) {
 searchChange(e);
 searchBlur(e);
}

function handleSubmit() {
 var f = document.getElementById('searchform');
 var t = document.getElementById('searchinput');
 var r = document.getElementById('real\_searchinput');

 //alert("handleSubmit "+ t.value);
 if(t.value.match(/review/)) {
 r.value = t.value;
 } else {
 //r.value = t.value + " -PackageReviewCategory -StackReviewCategory -M3Review -DocReview -ApiReview -HelpOn -BadContent -LocalSpellingWords";
 r.value = t.value + " -PackageReviewCategory -StackReviewCategory -DocReview -ApiReview";
 }
 //return validate(f);
}
//-->

|  |  |
| --- | --- |
| [ros.org](/ "/") | [About](http://www.ros.org/about-ros "http://www.ros.org/about-ros")
 |
 [Support](/Support "/Support")
 |
 [Discussion Forum](http://discourse.ros.org/ "http://discourse.ros.org/")
 |
 [Index](http://index.ros.org/ "http://index.ros.org/")
 |
 [Service Status](http://status.ros.org/ "http://status.ros.org/")
 |
 [Q&A answers.ros.org](http://answers.ros.org/ "http://answers.ros.org/") |
| [Documentation](/ "/")[Browse Software](https://index.ros.org/packages "https://index.ros.org/packages")[News](https://discourse.ros.org/c/general "https://discourse.ros.org/c/general")[Download](/ROS/Installation "/ROS/Installation") |

* [ROS](/ROS "/ROS")
* [Tutorials](/ROS/Tutorials "/ROS/Tutorials")
* [rosbuild](/ROS/Tutorials/rosbuild "/ROS/Tutorials/rosbuild")
* [NavigatingTheFilesystem](/ROS/Tutorials/rosbuild/NavigatingTheFilesystem "/ROS/Tutorials/rosbuild/NavigatingTheFilesystem")

#### ROS 2 Documentation

The ROS Wiki is for ROS 1. Are you using ROS 2 ([Humble](http://docs.ros.org/en/humble/ "http://docs.ros.org/en/humble/"), [Iron](http://docs.ros.org/en/iron/ "http://docs.ros.org/en/iron/"), or [Rolling](http://docs.ros.org/en/rolling/ "http://docs.ros.org/en/rolling/"))?   
[Check out the ROS 2 Project Documentation](http://docs.ros.org "http://docs.ros.org")  
Package specific documentation can be found on [index.ros.org](https://index.ros.org "https://index.ros.org")

# Wiki

* [Distributions](/Distributions "/Distributions")
* [ROS/Installation](/ROS/Installation "/ROS/Installation")
* [ROS/Tutorials](/ROS/Tutorials "/ROS/Tutorials")
* [RecentChanges](/RecentChanges "/RecentChanges")
* [NavigatingTheFilesystem](/ROS/Tutorials/rosbuild/NavigatingTheFilesystem "/ROS/Tutorials/rosbuild/NavigatingTheFilesystem")

# Page

* Immutable Page
* [Comments](# "#")
* [Info](/action/info/ROS/Tutorials/rosbuild/NavigatingTheFilesystem?action=info "/action/info/ROS/Tutorials/rosbuild/NavigatingTheFilesystem?action=info")
* [Attachments](/action/AttachFile/ROS/Tutorials/rosbuild/NavigatingTheFilesystem?action=AttachFile "/action/AttachFile/ROS/Tutorials/rosbuild/NavigatingTheFilesystem?action=AttachFile")
* More Actions:

Raw Text
Print View
Render as Docbook
Delete Cache
------------------------
Check Spelling
Like Pages
Local Site Map
------------------------
Rename Page
Copy Page
Delete Page
------------------------
My Pages
Subscribe User
------------------------
Remove Spam
Revert to this revision
Package Pages
Sync Pages
------------------------
CreatePdfDocument
Load
RawFile
Save
SlideShow

<!--// Init menu
actionsMenuInit('More Actions:');
//-->

# User

* [Login](/action/login/ROS/Tutorials/rosbuild/NavigatingTheFilesystem?action=login "/action/login/ROS/Tutorials/rosbuild/NavigatingTheFilesystem?action=login")

Contents1. [Prerequisites](#Prerequisites "#Prerequisites")
2. [Quick Overview of Filesystem Concepts](#Quick_Overview_of_Filesystem_Concepts "#Quick_Overview_of_Filesystem_Concepts")
3. [Filesystem Tools](#Filesystem_Tools "#Filesystem_Tools")
	1. [Using rospack and rosstack](#Using_rospack_and_rosstack "#Using_rospack_and_rosstack")
	2. [Using roscd](#Using_roscd "#Using_roscd")
		1. [Subdirectories](#Subdirectories "#Subdirectories")
	3. [Special cases for roscd](#Special_cases_for_roscd "#Special_cases_for_roscd")
		1. [roscd with no arguments](#roscd_with_no_arguments "#roscd_with_no_arguments")
		2. [roscd log](#roscd_log "#roscd_log")
	4. [Using rosls](#Using_rosls "#Using_rosls")
	5. [Tab Completion](#Tab_Completion "#Tab_Completion")
4. [Review](#Review "#Review")

## Prerequisites

For this tutorial we will inspect a package in ros-tutorials, please install it using 
```
$ sudo apt-get install ros-<distro>-ros-tutorials
```
Replace '<distro>' (including the '<>') with the name of your ROS distribution (e.g. hydro, groovy, electric, fuerte etc.) 
## Quick Overview of Filesystem Concepts

* [Packages](/Packages "/Packages"): Packages are the lowest level of ROS software organization. They can contain anything: libraries, tools, executables, etc.
* [Manifest](/Manifest "/Manifest"): A manifest is a description of a *package*. Its most important role is to define dependencies between *packages*.
* [Stacks](/Stacks "/Stacks"): Stacks are collections of *package*s that form a higher-level library.
* [Stack Manifest](/Stack%20Manifest "/Stack%20Manifest"): These are just like normal *manifests*, but for *stacks*.

When you look at the filesystem, it's easy to tell *packages* and *stacks* apart: * A package is a directory with a manifest.xml file.
* A stack is a directory with a stack.xml file.

![filesystem_layout.png](/ROS/Tutorials/rosbuild/NavigatingTheFilesystem?action=AttachFile&do=get&target=filesystem_layout.png "filesystem_layout.png") 
## Filesystem Tools

Code is spread across many ROS packages and stacks. Navigating with command-line tools such as ls and cd can be very tedious which is why ROS provides tools to help you. 
### Using rospack and rosstack

[rospack](/rospack "/rospack") and [rosstack](/rosstack "/rosstack") allow you to get information about packages and stacks. In this tutorial, we are only going to cover the find option, which returns the path to package or stack. Usage: 
```
$ rospack find [package_name]
$ rosstack find [stack_name]
```
Example: 
```
$ rospack find roscpp
```
Would return: * 
```
YOUR_INSTALL_PATH/share/roscpp
```

If, for example, you have used the [binary install](/fuerte/Installation/Ubuntu "/fuerte/Installation/Ubuntu") of [ROS Fuerte](/fuerte "/fuerte") on Ubuntu linux, you would see exactly: * 
```
/opt/ros/fuerte/share/roscpp
```

### Using roscd

roscd is part of the [rosbash](/rosbash "/rosbash") suite. It allows you to change directory ([cd](http://ss64.com/bash/cd.html "http://ss64.com/bash/cd.html")) directly to a package or a stack. Usage: 
```
$ roscd [locationname[/subdir]]
```
Run this example: 
```
$ roscd roscpp
```
To verify that we have changed to the roscpp package directory. Now let's print the working directory using the Unix command [pwd](http://ss64.com/bash/pwd.html "http://ss64.com/bash/pwd.html"): 
```
$ pwd
```
You should see: * 
```
YOUR_INSTALL_PATH/share/roscpp
```

You can see that YOUR\_INSTALL\_PATH/share/roscpp is the same path that rospack find gave in the previous example. Note that roscd, like other ROS tools, will *only* find ROS packages that are below the directories listed in your $ROS\_PACKAGE\_PATH. To see what is in your $ROS\_PACKAGE\_PATH, type: 
```
$ echo $ROS_PACKAGE_PATH
```
If you have not modified your $ROS\_PACKAGE\_PATH, you should see: * 
```
YOUR_INSTALL_PATH/share:YOUR_INSTALL_PATH/stacks
```

Similarly to other environment paths, you can add additional directories to your $ROS\_PACKAGE\_PATH, with each path separated by a colon ':' 
#### Subdirectories

roscd can also move to a subdirectory of a package or stack. Try: 
```
$ roscd roscpp/cmake
$ pwd
```
You should see: * 
```
YOUR_INSTALL_PATH/share/roscpp/cmake
```

### Special cases for roscd

There are a few special places you can tell roscd to go, that are not a package or stack. 
#### roscd with no arguments

roscd without an argument will take you to $ROS\_WORKSPACE. Try: 
```
$ roscd
$ pwd
```
You should see: * 
```
 /home/user/fuerte_workspace
```

Note: Prior to Fuerte, roscd would take you to $ROS\_ROOT. 
#### roscd log

roscd log will take you to the folder where ROS stores log files. Note that if you have not run any ROS programs yet, this will yield an error saying that it does not yet exist. If you have run some ROS program before, try: 
```
$ roscd log
```

### Using rosls

rosls is part of the [rosbash](/rosbash "/rosbash") suite. It allows you to [ls](http://ss64.com/bash/ls.html "http://ss64.com/bash/ls.html") directly in a package, stack, or common location by name rather than by package path. Usage: 
```
$ rosls [locationname[/subdir]]
```
Example: 
```
$ rosls roscpp_tutorials
```
Would return: * 
```
bin  cmake  manifest.xml  srv
```

### Tab Completion

It can get tedious to type out an entire package name. In the previous example, roscpp\_tutorials is a fairly long name. Luckily, some ROS tools support [TAB completion](http://en.wikipedia.org/wiki/Command_line_completion "http://en.wikipedia.org/wiki/Command_line_completion"). Start by typing: 
```
$ roscd roscpp_tut<<< now push the TAB key >>>
```
After pushing the **TAB** key, the command line should fill out the rest. * 
```
$ roscd roscpp_tutorials/
```

This works because roscpp\_tutorials is currently the only ROS package that starts with roscpp\_tut. Now try typing: 
```
$ roscd tur<<< now push the TAB key >>>
```
After pushing the **TAB** key, the command line should fill out as much as possible: * 
```
$ roscd turtle
```

However, in this case there are multiple packages that begin with turtle. Try typing **TAB** another time. This should display all the ROS packages that begin with turtle * 
```
  turtle_actionlib/  turtlesim/         turtle_tf/
```

On the command line you should still have * 
```
$ roscd turtle
```

Now type a s after turtle and then push **TAB** 
```
$ roscd turtles<<< now push the TAB key >>>
```
Since there is only one package that start with turtles, you should see: * 
```
$ roscd turtlesim/
```

## Review

You may have noticed a pattern with the naming of the ROS tools: * rospack = ros + pack(age)
* rosstack = ros + stack
* roscd = ros + cd
* rosls = ros + ls

This naming pattern holds for many of the ROS tools. 

Wiki: ROS/Tutorials/rosbuild/NavigatingTheFilesystem (last edited 2018-07-15 20:50:44 by [AustinHendrix](/AustinHendrix "AustinHendrix @ c-73-223-106-246.hsd1.ca.comcast.net[73.223.106.246]"))

Except where otherwise noted, the ROS wiki is licensed under the   
