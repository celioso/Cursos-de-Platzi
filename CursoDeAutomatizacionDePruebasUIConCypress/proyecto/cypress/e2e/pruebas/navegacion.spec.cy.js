describe("Navegacion",()=>{

    it("Navegar a nuestra primer pagina", ()=>{
        cy.visit("https://platzi.com/")
    })

    it("recargar pagina", ()=>{
        cy.reload()
    })

    it("Recargar pagina de forma forzada", ()=>{
        cy.visit("https://www.google.com/")
        cy.reload(true)
    })

    it("Navegar hacia atras", ()=>{
        cy.visit("https://www.google.com/")
        cy.visit("https://www.google.com/search?q=platzi&sca_esv=ff93aeab402caf98&sca_upv=1&sxsrf=ADLYWIJkMcLm-lmNgWyacjuWHWH9wtG2ZQ%3A1716241590848&source=hp&ei=tsRLZpLQMPKW4-EPnI6HoAk&iflsig=AL9hbdgAAAAAZkvSxv8iNltMKmFbHhMqKpB5Ad9CIjHs&ved=0ahUKEwiSg_mSmp2GAxVyyzgGHRzHAZQQ4dUDCBU&uact=5&oq=platzi&gs_lp=Egdnd3Mtd2l6IgZwbGF0emkyChAjGIAEGCcYigUyChAjGIAEGCcYigUyBBAjGCcyDhAuGIAEGLEDGNEDGMcBMggQABiABBixAzILEAAYgAQYsQMYgwEyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAESM4VUO8EWMUQcAF4AJABAZgBmQSgAaIKqgEHMC41LjUtMbgBA8gBAPgBAZgCBqACoAaoAgrCAgcQIxgnGOoCwgIREC4YgAQYsQMY0QMYgwEYxwHCAhEQLhiABBixAxiDARjUAhiKBcICDhAuGIAEGLEDGIMBGIoFwgIIEC4YgAQYsQOYAweSBwMxLjWgB41K&sclient=gws-wiz")
        //cy.go("back")
        cy.go(-1)
    })

    it.only("Navegar hacia adelante", ()=>{
        cy.visit("https://www.google.com/")
        cy.visit("https://www.google.com/search?q=platzi&sca_esv=ff93aeab402caf98&sca_upv=1&sxsrf=ADLYWIJkMcLm-lmNgWyacjuWHWH9wtG2ZQ%3A1716241590848&source=hp&ei=tsRLZpLQMPKW4-EPnI6HoAk&iflsig=AL9hbdgAAAAAZkvSxv8iNltMKmFbHhMqKpB5Ad9CIjHs&ved=0ahUKEwiSg_mSmp2GAxVyyzgGHRzHAZQQ4dUDCBU&uact=5&oq=platzi&gs_lp=Egdnd3Mtd2l6IgZwbGF0emkyChAjGIAEGCcYigUyChAjGIAEGCcYigUyBBAjGCcyDhAuGIAEGLEDGNEDGMcBMggQABiABBixAzILEAAYgAQYsQMYgwEyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAESM4VUO8EWMUQcAF4AJABAZgBmQSgAaIKqgEHMC41LjUtMbgBA8gBAPgBAZgCBqACoAaoAgrCAgcQIxgnGOoCwgIREC4YgAQYsQMY0QMYgwEYxwHCAhEQLhiABBixAxiDARjUAhiKBcICDhAuGIAEGLEDGIMBGIoFwgIIEC4YgAQYsQOYAweSBwMxLjWgB41K&sclient=gws-wiz")
        cy.go("back")
        //cy.go("forward")
        cy.go(1)

    it('Navegar hacia adelante en una pagina con chain command', () => {
        cy.visit('https://google.com');
        cy.visit('https://www.google.com/search?q=platzi&sxsrf=APq-WBsJmYoDdRVdbT5vkzyA6INN9o-OoA%3A1645072295957&source=hp&ei=p88NYtzpNpauytMPo56H6Aw&iflsig=AHkkrS4AAAAAYg3dt-lyynY6DU3aZCGsxCJKBESc0ZTy&ved=0ahUKEwic2c7u84X2AhUWl3IEHSPPAc0Q4dUDCAY&uact=5&oq=platzi&gs_lcp=Cgdnd3Mtd2l6EAMyBAgjECcyDgguEIAEELEDEMcBENEDMggIABCABBCxAzILCC4QgAQQxwEQrwEyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQ6BwgjEOoCECc6CwguEIAEELEDEIMBOggIABCxAxCDAToLCAAQgAQQsQMQgwE6CAguEIAEELEDOgYIIxAnEBM6BAgAEEM6BwgAELEDEEM6BwgAEMkDEEM6CgguEMcBEKMCEEM6DgguEIAEELEDEIMBENQCULcEWNgNYKYQaAFwAHgAgAGAAYgBxgWSAQMwLjaYAQCgAQGwAQo&sclient=gws-wiz').go('back').go('forward')
        });
    
    })
})