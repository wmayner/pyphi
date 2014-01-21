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
      docs:
        command: [
          "cd docs"
          "make html"
        ].join "&&"
        options:
          stdout: true
          failOnError: true

    watch:
      docs:
        files: [
          "<%= cfg.docDir %>/*"
          "<%= cfg.srcDir %>/*"
          "Gruntfile.coffee"
        ]
        tasks: ["shell:docs"]
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
    "watch:docs"
  ]
  grunt.registerTask "test", [
    "watch:test"
  ]
