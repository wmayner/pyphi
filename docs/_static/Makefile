docs = ..
docs_build = ../_build
docs_html = ../_build/html

watch-docs: build-docs
	watchmedo shell-command \
		--command='make build-docs' \
		--recursive --drop --ignore-directories \
		--patterns="*.css" \
		. $(docs)

build-docs:
	cd $(docs) && make html
	cp $(docs)/_static/*.css $(docs_html)/_static
	cp $(docs)/_static/*.png $(docs_html)/_static
