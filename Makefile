
.PHONY: test docs dist

src = pyphi
tests = test
docs = docs
docs_build = docs/_build
docs_html = docs/_build/html
benchmarks = benchmarks
dist_dir = dist

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

clean-docs:
	rm -rf $(docs_build)

build-docs:
	cd $(docs) && make html
	cp $(docs)/_static/*.css $(docs_html)/_static
	cp $(docs)/_static/*.png $(docs_html)/_static

open-docs:
	open $(docs_html)/index.html

upload-docs: build-docs
	cp -r $(docs_html) ../pyphi-docs
	cd ../pyphi-docs && git commit -a -m 'Update docs' && git push origin gh-pages

benchmark:
	cd $(benchmarks) && asv continuous develop

dist: build-dist
	twine upload dist/*

test-dist: build-dist
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

build-dist: clean-dist
	python setup.py sdist bdist_wheel
	python setup.py check -r -s

clean-dist:
	rm -r $(dist_dir)
