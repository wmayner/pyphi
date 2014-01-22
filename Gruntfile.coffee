module.exports = (grunt) ->
  grunt.initConfig
    cfg:
      docDir: "docs"
      srcDir: "cyphi"

    shell:
      test:
        command: "py.test"
        options:
          stdout: true
      buildDocs:
        command: [
          "cd docs"
          "make html"
        ].join "&&"
        options:
          stdout: true
          failOnError: true
      openDocs:
        command: "open docs/_build/html/index.html"

    watch:
      docs:
        files: [
          "<%= cfg.docDir %>/*"
          "<%= cfg.srcDir %>/*"
          "Gruntfile.coffee"
        ]
        tasks: ["shell:buildDocs"]
      test:
        files: ["<%= cfg.srcDir %>/*", "Gruntfile.coffee"]
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
    "watch:docs"
  ]
  grunt.registerTask "test", [
    "watch:test"
  ]
