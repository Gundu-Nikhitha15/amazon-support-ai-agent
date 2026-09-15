from src.RAG import process_message


# ============================================================
# AMAZONHELP AI SUPPORT AGENT
# ============================================================

print()
print("=" * 60)
print("AMAZONHELP AI SUPPORT AGENT")
print("=" * 60)

print()
print("Enter a customer support message.")
print("Type 'exit' to stop the agent.")

print()


while True:

    # --------------------------------------------------------
    # Get customer message
    # --------------------------------------------------------

    customer_message = input("Customer: ").strip()


    # --------------------------------------------------------
    # Exit
    # --------------------------------------------------------

    if customer_message.lower() in {"exit", "quit"}:

        print()
        print("Goodbye!")

        break


    # --------------------------------------------------------
    # Empty input
    # --------------------------------------------------------

    if not customer_message:

        print(
            "Please enter a customer message."
        )

        print()

        continue


    try:

        # ----------------------------------------------------
        # Process message through RAG pipeline
        # ----------------------------------------------------

        result = process_message(
            customer_message
        )


        # ----------------------------------------------------
        # Display intent
        # ----------------------------------------------------

        print()
        print("-" * 60)

        print(
            "Predicted Intent:"
        )

        print(
            result["predicted_intent"]
        )


        # ----------------------------------------------------
        # Display confidence
        # ----------------------------------------------------

        print()

        print(
            "Confidence:"
        )

        print(
            result["confidence"]
        )


        # ----------------------------------------------------
        # Display escalation decision
        # ----------------------------------------------------

        print()

        print(
            "Decision:"
        )

        print(
            result["decision"]
        )


        # ----------------------------------------------------
        # Display escalation reason
        # ----------------------------------------------------

        print()

        print(
            "Reason:"
        )

        print(
            result["escalation_reason"]
        )


        # ----------------------------------------------------
        # Display retrieval similarity
        # ----------------------------------------------------

        print()

        print(
            "Best Retrieval Similarity:"
        )

        print(
            result["best_similarity"]
        )


        # ----------------------------------------------------
        # Display response
        # ----------------------------------------------------

        print()

        print(
            "Response:"
        )

        print(
            result["generated_response"]
        )


        # ----------------------------------------------------
        # Display top historical case
        # ----------------------------------------------------

        retrieved = result["retrieved"]


        if not retrieved.empty:

            print()

            print(
                "Top Historical Case:"
            )


            top_case = retrieved.iloc[0]


            print()

            print(
                "Customer:",
                top_case["customer_message"]
            )


            print()

            print(
                "Amazon:",
                top_case["amazon_response"]
            )


        print()
        print("-" * 60)
        print()


    except Exception as error:

        print()

        print(
            "Error:",
            error
        )

        print()