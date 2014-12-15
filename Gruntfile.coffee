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
          "cp -r docs/_build/html github-docs"
          "git checkout gh-pages"
          "cp -r github-docs/* ."
          "git add ."
          "git commit -a -m 'Update docs'"
          "git push"
          "git checkout develop"
          "rm -r ./github-docs"
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
    "shell:buildDocs"
  ]
  grunt.registerTask "watch-docs", [
    "shell:openDocs"
    "shell:buildDocs"
    "shell:watchDocs"
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
