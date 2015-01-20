module.exports = (grunt) ->
  grunt.initConfig
    cfg:
      docDir: "docs"
      srcDir: "pyphi"
      testDir: "test"

    shell:
      test:
        command: "coverage run --source <%= cfg.srcDir %> -m py.test"
        options:
          stdout: true
      coverageHTML:
        command: "coverage html"
        options:
          stdout: true
      buildDocs:
        command: [
          "cd docs"
          "make html"
          "cp _static/* _build/html/_static"
        ].join "&&"
        options:
          stdout: true
          failOnError: true
      openDocs:
        command: "open docs/_build/html/index.html"
      uploadGithubDocs:
        command: [
          "cp -r docs/_build/html/* ../pyphi-docs"
          "cd ../pyphi-docs"
          "git commit -a -m 'Update docs'"
          "git push origin gh-pages"
          "cd ../pyphi"
        ].join "&&"
        options:
          stdout: true
          failOnError: true
      openCoverage:
        command: "open htmlcov/index.html"

    watch:
      docs:
        files: [
          "Gruntfile.coffee"
          "**/*.rst"
          "<%= cfg.srcDir %>/**/*"
          "<%= cfg.docDir %>/**/*"
          "!<%= cfg.docDir %>/_*/**/*"
          "<%= cfg.docDir %>/_static/**/*"
          "!**/pyphi.log"
          "!**/__pyphi_cache__"
          "!**/__pyphi_cache__.BACKUP"
          "!**/*.pyc"
        ]
        tasks: ["shell:buildDocs"]
      test:
        files: [
          "conftest.py"
          ".coveragerc"
          "pytest.ini"
          ".pythonrc.py"
          "<%= cfg.srcDir %>/**/*"
          "<%= cfg.testDir %>/**/*"
          "Gruntfile.coffee"
        ]
        tasks: ["shell:test", "shell:coverageHTML"]

  # Load NPM Tasks
  grunt.loadNpmTasks "grunt-contrib-watch"
  grunt.loadNpmTasks "grunt-shell"

  # Custom Tasks
  grunt.registerTask "default", [
    "watch:test"
  ]
  grunt.registerTask "docs", [
    "shell:buildDocs"
    "shell:openDocs"
  ]
  grunt.registerTask "watch-docs", [
    "shell:buildDocs"
    "shell:openDocs"
    "watch:docs"
  ]
  grunt.registerTask "upload-github-docs", [
    "shell:buildDocs"
    "shell:uploadGithubDocs"
  ]
  grunt.registerTask "test", [
    "shell:test"
    "shell:coverageHTML"
    "shell:openCoverage"
    "watch:test"
  ]
