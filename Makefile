.PHONY: test docs dist

src = pyphi
tests = test
docs = docs
docs_build = docs/_build
docs_html = docs/_build/html
benchmarks = benchmarks
dist_dir = dist
docs_port = 1337

test: coverage watch-tests

coverage:
	coverage run --source $(src) -m py.test
	coverage html
	open htmlcov/index.html

lint:
	pylint $(src)

watch-tests:
	watchmedo shell-command \
		--command='make coverage' \
		--recursive --drop --ignore-directories \
		--patterns="*.py;*.rst" $(src) $(tests) $(docs)
		# TODO: watch test config files

docs: build-docs

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

serve-docs: build-docs
	cd $(docs_html) && python -m http.server $(docs_port)

open-docs:
	open http://0.0.0.0:$(docs_port)

upload-docs: build-docs
	cp -r $(docs_html) ../pyphi-docs
	cd ../pyphi-docs && git commit -a -m 'Update docs' && git push origin gh-pages

benchmark:
	cd $(benchmarks) && asv continuous develop

check-dist:
	python setup.py check --strict

dist: build-dist check-dist
	twine upload $(dist_dir)/*

test-dist: build-dist check-dist
	twine upload --repository-url https://test.pypi.org/legacy/ $(dist_dir)/*

build-dist: clean-dist
	python setup.py sdist bdist_wheel --dist-dir=$(dist_dir)

clean-dist:
	rm -rf $(dist_dir)

clean:
	rm -rf **/__pycache__
