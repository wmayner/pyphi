
.PHONY: test docs

src = pyphi
tests = test
docs = docs
docs_html = docs/_build/html

test: coverage-test coverage-html coverage-open

watch-test:
	watchmedo shell-command \
		--command='make test' \
		--recursive --drop --ignore-directories \
		--patterns="*.py" $(src) $(tests)

coverage-test:
	coverage run --source $(src) -m py.test

coverage-html:
	coverage html

coverage-open:
	open htmlcov/index.html

docs: build-docs open-docs

watch-docs:
	watchmedo shell-command \
		--command='make docs' \
		--recursive --drop --ignore-directories \
		--patterns="*.py;*.rst" $(src) $(docs)

build-docs:
	cd $(docs) && make html
	cp $(docs)/_static/* $(docs_html)/_static

open-docs:
	open $(docs_html)/index.html

upload-docs: build-docs
	cp -r $(doc_html) ../pyphi-docs
	cd ../pyphi-docs && git commit -a -m 'Update docs' && git push origin gh-pages
