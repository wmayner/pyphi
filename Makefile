
.PHONY: test docs

src = pyphi
tests = test
docs = docs
docs_build = docs/_build
docs_html = docs/_build/html
benchmarks = benchmarks

test: coverage watch-tests

coverage:
	coverage run --source $(src) -m py.test
	coverage html
	open htmlcov/index.html

watch-tests:
	watchmedo shell-command \
		--command='make coverage' \
		--recursive --drop --ignore-directories \
		--patterns="*.py;*.rst" $(src) $(tests) $(docs)
		# TODO: watch test config files

docs: build-docs open-docs

watch-docs: docs
	watchmedo shell-command \
		--command='make build-docs' \
		--recursive --drop --ignore-directories \
		--patterns="*.py;*.rst" $(src) $(docs)

rm-docs:
	rm -rf $(docs_build)

build-docs:
	cd $(docs) && make html
	cp $(docs)/_static/* $(docs_html)/_static

open-docs:
	open $(docs_html)/index.html

upload-docs: build-docs
	cp -r $(docs_html) ../pyphi-docs
	cd ../pyphi-docs && git commit -a -m 'Update docs' && git push origin gh-pages

benchmark:
	cd $(benchmarks) && asv continuous develop
