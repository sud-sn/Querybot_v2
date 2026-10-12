"""The product's name, as the people who use it see it.

Pages, the translation catalogue and the AI's instructions carry the name as words; Python that puts a name
together (an email's header and sender, the API's title) takes it from here. Names a running server depends
on are not the product's name and stay as they were: the QUERYBOT_* settings, the querybot.db store, the
service, the Snowflake query tag the history reader leaves out, and the X-QueryBot-Export-ID header.
"""

PRODUCT_NAME = "Solviz"
