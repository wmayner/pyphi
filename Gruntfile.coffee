module.exports = (grunt) ->
  grunt.initConfig
    cfg:
      docDir: "docs"
      srcDir: "cyphi"
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
    "shell:openDocs"
    "watch:docs"
  ]
  grunt.registerTask "test", [
    "shell:test"
    "shell:coverageHTML"
    "shell:openCoverage"
    "watch:test"
  ]
