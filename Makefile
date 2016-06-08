
.PHONY: test docs

src = pyphi
tests = test
docs = docs
docs_build = docs/_build
docs_html = docs/_build/html


test: test-coverage coverage-html open-coverage

watch-test:
	watchmedo shell-command \
		--command='make test' \
		--recursive --drop --ignore-directories \
		--patterns="*.py" $(src) $(tests)

test-coverage:
	coverage run --source $(src) -m py.test

coverage-html:
	coverage html

open-coverage:
	open htmlcov/index.html

docs: build-docs open-docs

watch-docs:
	watchmedo shell-command \
		--command='make docs' \
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
