Detailed installation guide for Mac OS X
========================================

This is a step-by-step guide intended for those unfamiliar with Python or the
command-line (*a.k.a.* the “shell”).

A shell can be opened by opening a new tab in the Terminal app (located in
Utilities). Text that is `formatted like code` is meant to be copied and pasted
into the terminal (hit the Enter key to run the command).

The fist step is to install the versions of Python that we need. The most
convenient way of doing this is to use the OS X package manager
[Homebrew](http://brew.sh/). Install Homebrew by running this command:

```bash
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Now you should have access to the `brew` command. First, we need to install
Python 2 and 3. Using these so-called “brewed” Python versions, rather than the
version of Python that comes with your computer, will protect your computer's
Python version from unwanted changes that could interfere with other
applications.

```bash
brew install python python3
```

Then we need to ensure that the terminal “knows about” the newly-installed
Python versions:

```bash
brew link --overwrite python
brew link --overwrite python3
```

Now that we're using our shiny new Python versions, it is highly recommended to
set up a **virtual environment** in which to install PyPhi. Virtual
environments allow different projects to isolate their dependencies from one
another, so that they don't interact in unexpected ways. Please see [this
guide](http://docs.python-guide.org/en/latest/dev/virtualenvs/) for more
information.

To do this, you must install `virtualenv` and `virtualenvwrapper`, a [tool for
manipulating virtual
environments](http://virtualenvwrapper.readthedocs.org/en/latest/). Both of
those tools are available on [PyPI](https://pypi.python.org/pypi), the Python
package index, and can be installed with `pip`, the command-line utility for
installing and managing Python packages (`pip` was installed automatically with
the brewed Python):

```bash
pip install virtualenvwrapper
```

Now we need to edit your shell startup file. This is a file that runs
automatically every time you open a new shell (a new window or tab in the
Terminal app). This file should be in your home directory, though it will be
invisible in the Finder because the filename is preceded by a period. On most
Macs it is called `.bash_profile`. You can open this in a text editor by
running this command:

```bash
open -a TextEdit ~/.bash_profile
```

If this doesn't work because the file doesn't exist, then run `touch
~/.bash_profile` first.

Now, you'll add three lines to the shell startup file. These lines will set the
location where the virtual environments will live, the location of your
development project directories, and the location of the script installed with
this package, respectively. **Note:** The location of the script can be found
by running `which virtualenvwrapper.sh`.

The filepath after the equals sign on second line will different for everyone,
but here is an example:

```bash
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/dev
source /usr/local/bin/virtualenvwrapper.sh
```

After editing the startup file and saving it, open a new terminal shell by
opening a new tab or window (or just reload the startup file by running `source
~/.bash_profile`).

Now that `virtualenvwrapper` is fully installed, use it to create a Python 3
virtual environment, like so:

```bash
mkvirtualenv -p `which python3` <name_of_your_project>
```

The `` -p `which python3 ``\` option ensures that when the virtual environment
is activated, the commands `python` and `pip` will refer to their Python 3
counterparts.

The virtual environment should have been activated automatically after creating
it. It can be manually activated with `workon <name_of_your_project>`, and
deactivated with `deactivate`.

**Important:** Remember to activate the virtual environment *every time* you
begin working on your project. Also, note that the currently active virtual
environment is *not* associated with any particular folder; it is associated
with a terminal shell.

Finally, you can install PyPhi into your new virtual environment:

```bash
pip install pyphi
```

Congratulations, you've just installed PyPhi!

To play around with the software, ensure that you've activated the virtual
environment with `workon <name_of_your_project>`. Then run `python` to start a
Python 3 interpreter. Then, in the interpreter's command-line (which is
preceded by the `>>>` prompt), run

```python
import pyphi
```

Please see the documentation for some
[examples](http://pythonhosted.org/pyphi/#usage-and-examples) and information
on how to [configure](http://pythonhosted.org/pyphi/#configuration-optional)
it.
