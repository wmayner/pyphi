module.exports = (grunt) ->
  grunt.initConfig
    cfg:
      docDir: "docs"
      srcDir: "cyphi"
      testDir: "test"

    shell:
      test:
        command: "py.test --color yes"
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
        tasks: ["shell:test"]

  # Load NPM Tasks
  grunt.loadNpmTasks "grunt-contrib-watch"
  grunt.loadNpmTasks "grunt-shell"

  # Custom Tasks
  grunt.registerTask "default", [
    "watch:test"
  ]
  grunt.registerTask "docs", [
    "shell:openDocs"
    "shell:openCoverage"
    "watch:docs"
  ]
  grunt.registerTask "test", [
    "shell:test"
    "watch:test"
  ]
